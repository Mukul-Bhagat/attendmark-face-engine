import base64
import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import requests
from fastapi import Depends, FastAPI, Header, HTTPException, Response, status
from pydantic import BaseModel, Field

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None

try:
    import faiss  # type: ignore
except Exception:  # pragma: no cover
    faiss = None

try:
    from insightface.app import FaceAnalysis  # type: ignore
except Exception:  # pragma: no cover
    FaceAnalysis = None


APP_VERSION = '1.0.0'
DEFAULT_FACE_MODEL_NAME = 'buffalo_s'
DEFAULT_FACE_MODEL_INIT_MODE = 'background'
DEFAULT_FACE_MODEL_ALLOWED_MODULES = ['detection', 'recognition']
VALID_FACE_MODEL_INIT_MODES = {'background', 'lazy'}
DEFAULT_THRESHOLD = 0.72
BLUR_MIN_VARIANCE = float(os.getenv('FACE_BLUR_MIN_VARIANCE', '60'))
LIGHTING_MIN = float(os.getenv('FACE_LIGHTING_MIN', '0.16'))
LIGHTING_MAX = float(os.getenv('FACE_LIGHTING_MAX', '0.92'))
INDEX_DIR = Path(os.getenv('FACE_INDEX_DIR', './indexes')).resolve()
INDEX_DIR.mkdir(parents=True, exist_ok=True)

LOG_LEVEL = (os.getenv('LOG_LEVEL', 'INFO') or 'INFO').strip().upper()
logging.basicConfig(
    level=LOG_LEVEL,
    format='%(asctime)s %(levelname)s [%(name)s] %(message)s',
)
LOGGER = logging.getLogger('face_engine')


def _resolve_model_name() -> str:
    candidate = (os.getenv('FACE_MODEL_NAME', DEFAULT_FACE_MODEL_NAME) or DEFAULT_FACE_MODEL_NAME).strip()
    return candidate or DEFAULT_FACE_MODEL_NAME


def _resolve_model_init_mode() -> str:
    candidate = (os.getenv('FACE_MODEL_INIT_MODE', DEFAULT_FACE_MODEL_INIT_MODE) or DEFAULT_FACE_MODEL_INIT_MODE).strip().lower()
    if candidate in VALID_FACE_MODEL_INIT_MODES:
        return candidate
    LOGGER.warning(
        "Invalid FACE_MODEL_INIT_MODE=%s. Falling back to %s.",
        candidate,
        DEFAULT_FACE_MODEL_INIT_MODE,
    )
    return DEFAULT_FACE_MODEL_INIT_MODE


def _resolve_allowed_modules() -> List[str]:
    raw = os.getenv(
        'FACE_MODEL_ALLOWED_MODULES',
        ','.join(DEFAULT_FACE_MODEL_ALLOWED_MODULES),
    ) or ''
    modules = [module.strip() for module in raw.split(',') if module.strip()]
    return modules if modules else list(DEFAULT_FACE_MODEL_ALLOWED_MODULES)


def _resolve_det_size() -> int:
    raw = os.getenv('FACE_MODEL_DET_SIZE', '320')
    try:
        value = int(raw)
    except Exception:
        value = 320
    return max(160, min(1024, value))


FACE_MODEL_DET_SIZE = _resolve_det_size()
FACE_MODEL_NAME = _resolve_model_name()
FACE_MODEL_INIT_MODE = _resolve_model_init_mode()
FACE_MODEL_ALLOWED_MODULES = _resolve_allowed_modules()


class EnrollRequest(BaseModel):
    organizationId: str
    userId: str
    imageUrls: List[str] = Field(min_length=1)


class MatchByClassRequest(BaseModel):
    organizationId: str
    classId: str
    imageBase64: str
    expectedUserId: str
    threshold: float = Field(default=DEFAULT_THRESHOLD, ge=0.3, le=0.99)


class UpsertRequest(BaseModel):
    organizationId: str
    classId: str
    userId: str
    embedding: List[float]


class RemoveRequest(BaseModel):
    organizationId: str
    classId: str
    userId: str


class QualityCheckRequest(BaseModel):
    imageBase64: str


class WarmupRequest(BaseModel):
    organizationId: str
    classIds: List[str] = Field(default_factory=list)


class ClassIndexState:
    def __init__(self) -> None:
        self.user_vectors: Dict[str, np.ndarray] = {}
        self.index = None
        self.id_order: List[str] = []


class FaceEngine:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._class_indexes: Dict[str, ClassIndexState] = {}
        self._model = None
        self._insightface_ready = False
        self._model_error: Optional[str] = None
        self._model_init_started = False
        self._model_init_in_progress = False
        self._model_initialized_at: Optional[int] = None
        self._model_init_lock = threading.Lock()
        self._load_indexes()
        self._log_runtime_config()
        self.ensure_model_initialized(async_init=FACE_MODEL_INIT_MODE != 'lazy')

    def _log_runtime_config(self) -> None:
        LOGGER.info(
            (
                "Face engine startup config: model=%s det_size=%s init_mode=%s "
                "allowed_modules=%s index_dir=%s thread_caps={OMP=%s,OPENBLAS=%s,MKL=%s,NUMEXPR=%s}"
            ),
            FACE_MODEL_NAME,
            FACE_MODEL_DET_SIZE,
            FACE_MODEL_INIT_MODE,
            ','.join(FACE_MODEL_ALLOWED_MODULES),
            INDEX_DIR,
            os.getenv('OMP_NUM_THREADS', 'unset'),
            os.getenv('OPENBLAS_NUM_THREADS', 'unset'),
            os.getenv('MKL_NUM_THREADS', 'unset'),
            os.getenv('NUMEXPR_NUM_THREADS', 'unset'),
        )

    def _set_model_failure(self, message: str) -> None:
        with self._lock:
            self._model = None
            self._insightface_ready = False
            self._model_error = message
            self._model_initialized_at = None

    def _set_model_ready(self, model) -> None:
        with self._lock:
            self._model = model
            self._insightface_ready = True
            self._model_error = None
            self._model_initialized_at = int(time.time())

    def _init_model(self) -> None:
        if FaceAnalysis is None or cv2 is None:
            if FaceAnalysis is None and cv2 is None:
                self._set_model_failure('InsightFace and OpenCV imports failed.')
            elif FaceAnalysis is None:
                self._set_model_failure('InsightFace import failed.')
            else:
                self._set_model_failure('OpenCV import failed.')
            return

        try:
            kwargs = {
                'name': FACE_MODEL_NAME,
                'providers': ['CPUExecutionProvider'],
            }
            if FACE_MODEL_ALLOWED_MODULES:
                kwargs['allowed_modules'] = FACE_MODEL_ALLOWED_MODULES
            model = FaceAnalysis(**kwargs)

            model.prepare(ctx_id=0, det_size=(FACE_MODEL_DET_SIZE, FACE_MODEL_DET_SIZE))
            self._set_model_ready(model)
            LOGGER.info(
                "InsightFace model initialized successfully: model=%s det_size=%s allowed_modules=%s",
                FACE_MODEL_NAME,
                FACE_MODEL_DET_SIZE,
                ','.join(FACE_MODEL_ALLOWED_MODULES),
            )
        except TypeError as exc:
            self._set_model_failure(
                (
                    "FaceAnalysis init failed with explicit allowed_modules. "
                    f"model={FACE_MODEL_NAME} allowed_modules={FACE_MODEL_ALLOWED_MODULES} error={exc}"
                )
            )
            LOGGER.exception('Failed to initialize InsightFace model: invalid constructor arguments')
        except Exception as exc:
            self._set_model_failure(str(exc))
            LOGGER.exception('Failed to initialize InsightFace model')

    def _init_model_worker(self) -> None:
        try:
            self._init_model()
        finally:
            with self._model_init_lock:
                self._model_init_in_progress = False

    def ensure_model_initialized(self, async_init: bool = True) -> None:
        with self._model_init_lock:
            if self._insightface_ready or self._model_init_in_progress:
                return
            self._model_init_started = True
            self._model_init_in_progress = True

        if async_init:
            threading.Thread(
                target=self._init_model_worker,
                name='insightface-init',
                daemon=True,
            ).start()
            return

        self._init_model_worker()

    def get_model_state(self) -> Dict[str, object]:
        with self._lock:
            if self._insightface_ready:
                state = 'ready'
            elif self._model_init_in_progress:
                state = 'initializing'
            elif self._model_error:
                state = 'failed'
            elif self._model_init_started:
                state = 'starting'
            else:
                state = 'idle'

            return {
                'state': state,
                'ready': self._insightface_ready,
                'error': self._model_error,
                'modelName': FACE_MODEL_NAME,
                'detSize': FACE_MODEL_DET_SIZE,
                'initMode': FACE_MODEL_INIT_MODE,
                'initializedAt': self._model_initialized_at,
                'allowedModules': FACE_MODEL_ALLOWED_MODULES,
            }

    def _key(self, organization_id: str, class_id: str) -> str:
        return f"org_{organization_id}_class_{class_id}"

    def _index_path(self, key: str) -> Path:
        return INDEX_DIR / f"{key}.index"

    def _meta_path(self, key: str) -> Path:
        return INDEX_DIR / f"{key}.meta.json"

    def _normalize_vector(self, vector: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(vector)
        if norm == 0:
            return vector
        return vector / norm

    def _decode_image_bytes(self, image_bytes: bytes):
        if cv2 is None:
            return None
        arr = np.frombuffer(image_bytes, dtype=np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)

    def _compute_quality(self, image) -> Dict[str, float]:
        if cv2 is None or image is None:
            return {'blurScore': 100.0, 'lightingScore': 0.5}

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        lighting_score = float(np.mean(gray) / 255.0)
        return {'blurScore': blur_score, 'lightingScore': lighting_score}

    def _validate_quality(self, quality: Dict[str, float]) -> Optional[str]:
        blur = quality.get('blurScore', 0.0)
        lighting = quality.get('lightingScore', 0.0)
        if blur < BLUR_MIN_VARIANCE:
            return 'BLURRY_IMAGE'
        if lighting < LIGHTING_MIN:
            return 'LOW_LIGHTING'
        if lighting > LIGHTING_MAX:
            return 'LOW_LIGHTING'
        return None

    def _extract_embedding(self, image_bytes: bytes) -> Tuple[Optional[np.ndarray], Dict[str, float], str, int]:
        image = self._decode_image_bytes(image_bytes)
        quality = self._compute_quality(image)

        if not self._insightface_ready or self._model is None:
            self.ensure_model_initialized(async_init=True)
            return None, quality, 'ENGINE_UNAVAILABLE', 0

        if image is None:
            return None, quality, 'INVALID_IMAGE', 0

        faces = self._model.get(image)
        face_count = len(faces)
        if face_count == 0:
            return None, quality, 'NO_FACE_DETECTED', 0
        if face_count > 1:
            return None, quality, 'MULTIPLE_FACES', face_count

        quality_error = self._validate_quality(quality)
        if quality_error:
            return None, quality, quality_error, 1

        embedding = np.array(faces[0].embedding, dtype=np.float32)
        return self._normalize_vector(embedding), quality, 'ok', 1

    def _fetch_image_bytes(self, url: str) -> bytes:
        response = requests.get(url, timeout=6)
        response.raise_for_status()
        return response.content

    def enroll_from_urls(self, organization_id: str, user_id: str, image_urls: List[str]) -> dict:
        accepted = []
        rejected = []
        vectors = []
        blur_scores: List[float] = []
        lighting_scores: List[float] = []

        for url in image_urls:
            try:
                image_bytes = self._fetch_image_bytes(url)
                vector, quality, reason, _face_count = self._extract_embedding(image_bytes)
                if vector is None:
                    rejected.append({'url': url, 'reason': reason})
                    continue

                accepted.append({
                    'url': url,
                    'blurScore': quality.get('blurScore'),
                    'lightingScore': quality.get('lightingScore'),
                })
                vectors.append(vector)
                if 'blurScore' in quality:
                    blur_scores.append(float(quality['blurScore']))
                if 'lightingScore' in quality:
                    lighting_scores.append(float(quality['lightingScore']))
            except Exception as exc:
                rejected.append({'url': url, 'reason': f'FETCH_OR_PROCESS_ERROR: {str(exc)}'})

        if len(vectors) == 0:
            engine_unavailable = any(
                str(entry.get('reason', '')).upper().startswith('ENGINE_UNAVAILABLE')
                for entry in rejected
            )
            if engine_unavailable:
                raise HTTPException(
                    status_code=503,
                    detail='FACE_ENGINE_UNAVAILABLE',
                )
            raise HTTPException(status_code=422, detail='No valid enrollment image. Ensure one clear face per image.')

        matrix = np.vstack(vectors)
        avg_vector = np.mean(matrix, axis=0)
        avg_vector = self._normalize_vector(avg_vector)

        quality_summary = {
            'acceptedCount': len(accepted),
            'rejectedCount': len(rejected),
            'blurScoreAvg': float(np.mean(blur_scores)) if blur_scores else None,
            'lightingScoreAvg': float(np.mean(lighting_scores)) if lighting_scores else None,
            'notes': 'insightface',
        }

        return {
            'embedding': avg_vector.astype(np.float32).tolist(),
            'acceptedImages': accepted,
            'rejectedImages': rejected,
            'qualitySummary': quality_summary,
        }

    def _ensure_state(self, key: str) -> ClassIndexState:
        state = self._class_indexes.get(key)
        if state is None:
            state = ClassIndexState()
            self._class_indexes[key] = state
        return state

    def _rebuild_class_index(self, key: str) -> None:
        state = self._ensure_state(key)
        state.id_order = sorted(state.user_vectors.keys())

        if len(state.id_order) == 0:
            state.index = None
            self._persist_state(key)
            return

        matrix = np.vstack([state.user_vectors[user_id] for user_id in state.id_order]).astype(np.float32)
        if faiss is None:
            state.index = matrix
        else:
            dim = matrix.shape[1]
            index = faiss.IndexFlatIP(dim)
            index.add(matrix)
            state.index = index

        self._persist_state(key)

    def _persist_state(self, key: str) -> None:
        state = self._ensure_state(key)
        payload = {
            'ids': state.id_order,
            'vectors': [state.user_vectors[user_id].astype(np.float32).tolist() for user_id in state.id_order],
        }
        self._meta_path(key).write_text(json.dumps(payload), encoding='utf-8')

        if faiss is not None and state.index is not None:
            faiss.write_index(state.index, str(self._index_path(key)))
        else:
            if self._index_path(key).exists():
                self._index_path(key).unlink()

    def _load_state_from_meta(self, key: str, meta_path: Path) -> bool:
        try:
            payload = json.loads(meta_path.read_text(encoding='utf-8'))
            ids = payload.get('ids') or []
            vectors = payload.get('vectors') or []
            if len(ids) != len(vectors):
                return False

            state = self._ensure_state(key)
            state.user_vectors = {}
            for user_id, vector in zip(ids, vectors):
                np_vec = np.array(vector, dtype=np.float32)
                state.user_vectors[str(user_id)] = self._normalize_vector(np_vec)
            self._rebuild_class_index(key)
            return True
        except Exception:
            return False

    def _load_indexes(self) -> None:
        for meta_path in INDEX_DIR.glob('*.meta.json'):
            key = meta_path.stem.replace('.meta', '')
            self._load_state_from_meta(key, meta_path)

    def get_index_stats(self) -> Dict[str, int]:
        with self._lock:
            class_index_count = len(self._class_indexes)
            non_empty_class_index_count = sum(
                1 for state in self._class_indexes.values() if len(state.user_vectors) > 0
            )
        return {
            'classIndexCount': class_index_count,
            'nonEmptyClassIndexCount': non_empty_class_index_count,
        }

    def upsert_embedding(self, organization_id: str, class_id: str, user_id: str, embedding: List[float]) -> None:
        key = self._key(organization_id, class_id)
        with self._lock:
            state = self._ensure_state(key)
            vector = np.array(embedding, dtype=np.float32)
            state.user_vectors[user_id] = self._normalize_vector(vector)
            self._rebuild_class_index(key)

    def remove_embedding(self, organization_id: str, class_id: str, user_id: str) -> None:
        key = self._key(organization_id, class_id)
        with self._lock:
            state = self._ensure_state(key)
            if user_id in state.user_vectors:
                del state.user_vectors[user_id]
            self._rebuild_class_index(key)

    def quality_check(self, image_base64: str) -> dict:
        try:
            image_bytes = base64.b64decode(image_base64, validate=True)
        except Exception:
            raise HTTPException(status_code=400, detail='Invalid base64 image payload')

        vector, quality, reason, face_count = self._extract_embedding(image_bytes)
        return {
            'ok': vector is not None,
            'reason': None if vector is not None else reason,
            'faceCount': face_count,
            'quality': {
                'blurScore': quality.get('blurScore'),
                'lightingScore': quality.get('lightingScore'),
            },
        }

    def warmup_class_indexes(self, organization_id: str, class_ids: List[str]) -> dict:
        warmed = 0
        warmed_keys: List[str] = []

        with self._lock:
            if not class_ids:
                return {
                    'ok': True,
                    'warmedCount': len(self._class_indexes),
                    'warmedKeys': sorted(list(self._class_indexes.keys())),
                }

            for class_id in class_ids:
                key = self._key(organization_id, class_id)
                if key in self._class_indexes:
                    warmed += 1
                    warmed_keys.append(key)
                    continue

                meta_path = self._meta_path(key)
                if meta_path.exists() and self._load_state_from_meta(key, meta_path):
                    warmed += 1
                    warmed_keys.append(key)
                    continue

                self._ensure_state(key)
                warmed += 1
                warmed_keys.append(key)

        return {
            'ok': True,
            'warmedCount': warmed,
            'warmedKeys': warmed_keys,
        }

    def match_by_class(
        self,
        organization_id: str,
        class_id: str,
        image_base64: str,
        expected_user_id: str,
        threshold: float,
    ) -> dict:
        try:
            image_bytes = base64.b64decode(image_base64, validate=True)
        except Exception:
            raise HTTPException(status_code=400, detail='Invalid base64 image payload')

        vector, quality, reason, face_count = self._extract_embedding(image_bytes)
        if vector is None:
            return {
                'matched': False,
                'reason': reason,
                'quality': {
                    'faceCount': face_count,
                    'blurScore': quality.get('blurScore'),
                    'lightingScore': quality.get('lightingScore'),
                },
            }

        key = self._key(organization_id, class_id)
        state = self._ensure_state(key)
        if len(state.user_vectors) == 0:
            return {
                'matched': False,
                'reason': 'NOT_IN_CLASS',
                'quality': {
                    'faceCount': 1,
                    'blurScore': quality.get('blurScore'),
                    'lightingScore': quality.get('lightingScore'),
                },
            }

        if faiss is None or state.index is None:
            scores = []
            for user_id in state.id_order:
                stored = state.user_vectors[user_id]
                score = float(np.dot(vector, stored))
                scores.append((score, user_id))
            scores.sort(key=lambda item: item[0], reverse=True)
            top_confidence, top_user_id = scores[0]
        else:
            query = np.expand_dims(vector.astype(np.float32), axis=0)
            distances, indices = state.index.search(query, k=1)
            idx = int(indices[0][0])
            top_confidence = float(distances[0][0]) if idx >= 0 else 0.0
            top_user_id = state.id_order[idx] if idx >= 0 else None

        matched = top_user_id is not None and top_confidence >= threshold and top_user_id == expected_user_id
        return {
            'matched': matched,
            'matchedUserId': top_user_id,
            'confidence': float(max(0.0, min(1.0, top_confidence))),
            'distance': float(1.0 - top_confidence),
            'reason': None if matched else 'LOW_CONFIDENCE',
            'quality': {
                'faceCount': 1,
                'blurScore': quality.get('blurScore'),
                'lightingScore': quality.get('lightingScore'),
            },
        }


engine = FaceEngine()
app = FastAPI(title='AttendMark Face Engine', version=APP_VERSION)


def auth_guard(x_internal_api_key: Optional[str] = Header(default=None)) -> None:
    expected = os.getenv('FACE_ENGINE_API_KEY') or os.getenv('INTERNAL_API_KEY')
    if expected and x_internal_api_key != expected:
        raise HTTPException(status_code=401, detail='Invalid internal API key')


@app.get('/healthz')
def healthz(response: Response) -> dict:
    stats = engine.get_index_stats()
    model_state = engine.get_model_state()
    model_ready = bool(model_state['ready'])
    healthy = model_ready and faiss is not None

    if not healthy:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    return {
        'ok': healthy,
        'version': APP_VERSION,
        'insightface': model_ready,
        'faiss': faiss is not None,
        'modelError': model_state['error'],
        'modelState': model_state['state'],
        'modelName': model_state['modelName'],
        'modelDetSize': model_state['detSize'],
        'modelInitMode': model_state['initMode'],
        'modelAllowedModules': model_state['allowedModules'],
        'modelInitializedAt': model_state['initializedAt'],
        'indexesDir': str(INDEX_DIR),
        'classIndexCount': stats['classIndexCount'],
        'nonEmptyClassIndexCount': stats['nonEmptyClassIndexCount'],
        'timestamp': int(time.time()),
    }


@app.post('/v1/enroll', dependencies=[Depends(auth_guard)])
def enroll(payload: EnrollRequest) -> dict:
    return engine.enroll_from_urls(
        organization_id=payload.organizationId,
        user_id=payload.userId,
        image_urls=payload.imageUrls,
    )


@app.post('/v1/match/class', dependencies=[Depends(auth_guard)])
def match_class(payload: MatchByClassRequest) -> dict:
    return engine.match_by_class(
        organization_id=payload.organizationId,
        class_id=payload.classId,
        image_base64=payload.imageBase64,
        expected_user_id=payload.expectedUserId,
        threshold=payload.threshold,
    )


@app.post('/v1/index/upsert', dependencies=[Depends(auth_guard)])
def index_upsert(payload: UpsertRequest) -> dict:
    engine.upsert_embedding(
        organization_id=payload.organizationId,
        class_id=payload.classId,
        user_id=payload.userId,
        embedding=payload.embedding,
    )
    return {'ok': True}


@app.post('/v1/index/remove', dependencies=[Depends(auth_guard)])
def index_remove(payload: RemoveRequest) -> dict:
    engine.remove_embedding(
        organization_id=payload.organizationId,
        class_id=payload.classId,
        user_id=payload.userId,
    )
    return {'ok': True}


@app.post('/v1/quality-check', dependencies=[Depends(auth_guard)])
def quality_check(payload: QualityCheckRequest) -> dict:
    return engine.quality_check(payload.imageBase64)


@app.post('/v1/index/warmup', dependencies=[Depends(auth_guard)])
def index_warmup(payload: WarmupRequest) -> dict:
    return engine.warmup_class_indexes(payload.organizationId, payload.classIds)
