import io
import logging
import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, List, Optional, Sequence, Tuple

from PIL import Image

log = logging.getLogger(__name__)

EMBEDDING_BASE_CLIP = "clip"
EMBEDDING_BASE_PLACE = "place"

DEFAULT_CLIP_MODEL = os.getenv("GEOSPY_CLIP_MODEL", "ViT-B-32")
DEFAULT_CLIP_PRETRAINED = os.getenv("GEOSPY_CLIP_PRETRAINED", "laion2b_s34b_b79k")
DEFAULT_CLIP_VERSION = os.getenv("GEOSPY_CLIP_VERSION", "open_clip")

DEFAULT_PLACE_RUNTIME = os.getenv(
    "GEOSPY_PLACE_RUNTIME",
    "open_clip",
).strip().lower()
DEFAULT_PLACE_CLIP_MODEL = os.getenv(
    "GEOSPY_PLACE_CLIP_MODEL",
    "ViT-B-16",
).strip()
DEFAULT_PLACE_CLIP_PRETRAINED = os.getenv(
    "GEOSPY_PLACE_CLIP_PRETRAINED",
    "laion2b_s34b_b88k",
).strip()
DEFAULT_PLACE_MODEL = os.getenv(
    "GEOSPY_PLACE_MODEL_NAME",
    "",
).strip()
DEFAULT_PLACE_VERSION = os.getenv(
    "GEOSPY_PLACE_MODEL_VERSION",
    "open_clip_place",
)
DEFAULT_PLACE_TRUST_REMOTE_CODE = (
    os.getenv("GEOSPY_PLACE_TRUST_REMOTE_CODE", "0").strip().lower()
    in {"1", "true", "yes", "on"}
)
DEFAULT_HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")


def _is_enabled(raw_value: str, default: bool = True) -> bool:
    value = str(raw_value).strip().lower()
    if not value:
        return default
    return value in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class RetrievalModelConfig:
    model_id: str
    model_name: str
    pretrained: str
    model_version: str
    weight: float
    embedding_base: str
    runtime: str
    trust_remote_code: bool = False


def _build_retrieval_model_configs() -> List[RetrievalModelConfig]:
    configs: List[RetrievalModelConfig] = []
    clip_enabled = _is_enabled(os.getenv("GEOSPY_CLIP_MODEL_ENABLED", "1"), default=True)
    if clip_enabled:
        configs.append(
            RetrievalModelConfig(
                model_id="clip",
                model_name=DEFAULT_CLIP_MODEL,
                pretrained=DEFAULT_CLIP_PRETRAINED,
                model_version=DEFAULT_CLIP_VERSION,
                weight=max(0.0, float(os.getenv("GEOSPY_RETRIEVAL_PRIMARY_WEIGHT", "1.0"))),
                embedding_base=EMBEDDING_BASE_CLIP,
                runtime="open_clip",
            )
        )

    place_enabled = _is_enabled(
        os.getenv("GEOSPY_PLACE_MODEL_ENABLED", "1"),
        default=True,
    )
    if place_enabled:
        place_runtime = DEFAULT_PLACE_RUNTIME or "open_clip"
        if place_runtime not in {"open_clip", "hf_transformers"}:
            raise RuntimeError(
                "GEOSPY_PLACE_RUNTIME must be one of: open_clip, hf_transformers."
            )
        if place_runtime == "open_clip":
            if not DEFAULT_PLACE_CLIP_MODEL or not DEFAULT_PLACE_CLIP_PRETRAINED:
                raise RuntimeError(
                    "GEOSPY_PLACE_RUNTIME=open_clip requires "
                    "GEOSPY_PLACE_CLIP_MODEL and GEOSPY_PLACE_CLIP_PRETRAINED."
                )
            configs.append(
                RetrievalModelConfig(
                    model_id="place",
                    model_name=DEFAULT_PLACE_CLIP_MODEL,
                    pretrained=DEFAULT_PLACE_CLIP_PRETRAINED,
                    model_version=DEFAULT_PLACE_VERSION,
                    weight=max(
                        0.0,
                        float(
                            os.getenv(
                                "GEOSPY_RETRIEVAL_PLACE_WEIGHT",
                                "1.0",
                            )
                        ),
                    ),
                    embedding_base=EMBEDDING_BASE_PLACE,
                    runtime="open_clip",
                )
            )
        else:
            if not DEFAULT_PLACE_MODEL:
                raise RuntimeError(
                    "GEOSPY_PLACE_RUNTIME=hf_transformers requires GEOSPY_PLACE_MODEL_NAME."
                )
            configs.append(
                RetrievalModelConfig(
                    model_id="place",
                    model_name=DEFAULT_PLACE_MODEL,
                    pretrained="",
                    model_version=DEFAULT_PLACE_VERSION,
                    weight=max(
                        0.0,
                        float(
                            os.getenv(
                                "GEOSPY_RETRIEVAL_PLACE_WEIGHT",
                                "1.0",
                            )
                        ),
                    ),
                    embedding_base=EMBEDDING_BASE_PLACE,
                    runtime="hf_transformers",
                    trust_remote_code=DEFAULT_PLACE_TRUST_REMOTE_CODE,
                )
            )

    deduped: List[RetrievalModelConfig] = []
    seen = set()
    for cfg in configs:
        key = (
            cfg.model_id,
            cfg.model_name,
            cfg.pretrained,
            cfg.model_version,
            cfg.embedding_base,
            cfg.runtime,
        )
        if key in seen or cfg.weight <= 0:
            continue
        seen.add(key)
        deduped.append(cfg)
    return deduped


def _import_clip_runtime():
    try:
        import open_clip
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "CLIP dependencies are missing. Install open-clip-torch and torch."
        ) from exc
    return torch, open_clip


def _import_hf_runtime():
    try:
        import torch
        from transformers import AutoImageProcessor, AutoModel
    except ImportError as exc:
        raise RuntimeError(
            "Place-model dependencies are missing. Install transformers and torch."
        ) from exc
    return torch, AutoImageProcessor, AutoModel


class ClipEmbedder:
    def __init__(
        self,
        model_id: str = "clip",
        model_name: str = DEFAULT_CLIP_MODEL,
        pretrained: str = DEFAULT_CLIP_PRETRAINED,
        model_version: str = DEFAULT_CLIP_VERSION,
        weight: float = 1.0,
    ):
        self.model_id = model_id
        self.model_name = model_name
        self.pretrained = pretrained
        self.model_version = model_version
        self.weight = float(weight)
        self.embedding_base = EMBEDDING_BASE_CLIP
        self.runtime = "open_clip"
        self._torch = None
        self._model = None
        self._preprocess = None
        self._device = "cpu"
        self._hf_token = DEFAULT_HF_TOKEN
        self._query_adapter = None
        self._query_adapter_loaded = False

    def _ensure_loaded(self):
        if self._model is not None and self._preprocess is not None:
            return
        torch, open_clip = _import_clip_runtime()
        self._torch = torch
        if torch.cuda.is_available():
            self._device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self._device = "mps"
        else:
            self._device = "cpu"
        model, _, preprocess = open_clip.create_model_and_transforms(
            self.model_name, pretrained=self.pretrained
        )
        model.eval()
        model.to(self._device)
        self._model = model
        self._preprocess = preprocess

    def _get_query_adapter_path(self) -> str:
        model_specific = os.getenv(
            f"GEOSPY_RETRIEVAL_QUERY_ADAPTER_{self.model_id.upper()}_PATH", ""
        ).strip()
        if model_specific:
            return model_specific
        return os.getenv("GEOSPY_RETRIEVAL_QUERY_ADAPTER_PATH", "").strip()

    def _ensure_query_adapter_loaded(self):
        if self._query_adapter_loaded:
            return
        self._query_adapter_loaded = True
        adapter_path = self._get_query_adapter_path()
        if not adapter_path:
            return
        if not os.path.exists(adapter_path):
            log.warning(
                "Query adapter path does not exist model_id=%s path=%s",
                self.model_id,
                adapter_path,
            )
            return
        try:
            payload = self._torch.load(adapter_path, map_location=self._device)
            if isinstance(payload, dict):
                if "state_dict" in payload and isinstance(payload["state_dict"], dict):
                    state_dict = payload["state_dict"]
                elif "weight" in payload:
                    state_dict = {"weight": payload["weight"]}
                else:
                    state_dict = payload
            else:
                raise RuntimeError("Unsupported adapter checkpoint format")
            dim = int(self.embedding_dim)
            adapter = self._torch.nn.Linear(dim, dim, bias=False).to(self._device)
            adapter.load_state_dict(state_dict, strict=True)
            adapter.eval()
            self._query_adapter = adapter
        except Exception as exc:
            log.warning(
                "Failed to load query adapter model_id=%s path=%s error=%s",
                self.model_id,
                adapter_path,
                exc,
            )
            self._query_adapter = None

    def _apply_query_adapter(self, features):
        self._ensure_query_adapter_loaded()
        if self._query_adapter is None:
            return features
        transformed = self._query_adapter(features)
        return transformed / transformed.norm(dim=-1, keepdim=True).clamp(min=1e-12)

    @property
    def embedding_dim(self) -> int:
        self._ensure_loaded()
        value = getattr(self._model.visual, "output_dim", None)
        if value is None:
            raise RuntimeError("Unable to read CLIP embedding dimension")
        return int(value)

    def encode_image_bytes(self, image_bytes: bytes) -> List[float]:
        self._ensure_loaded()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        tensor = self._preprocess(image).unsqueeze(0).to(self._device)
        with self._torch.no_grad():
            features = self._model.encode_image(tensor)
            features = features / features.norm(dim=-1, keepdim=True).clamp(min=1e-12)
            features = self._apply_query_adapter(features)
        return features[0].detach().cpu().float().tolist()

    def encode_image_bytes_batch(self, image_bytes_batch: List[bytes]) -> List[List[float]]:
        self._ensure_loaded()
        if not image_bytes_batch:
            return []
        tensors = []
        for image_bytes in image_bytes_batch:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            tensors.append(self._preprocess(image))
        batch = self._torch.stack(tensors, dim=0).to(self._device)
        with self._torch.no_grad():
            features = self._model.encode_image(batch)
            features = features / features.norm(dim=-1, keepdim=True).clamp(min=1e-12)
            features = self._apply_query_adapter(features)
        return features.detach().cpu().float().tolist()


class PlaceEmbedder:
    def __init__(
        self,
        model_id: str = "place",
        model_name: str = DEFAULT_PLACE_MODEL,
        pretrained: str = "",
        model_version: str = DEFAULT_PLACE_VERSION,
        weight: float = 1.0,
        trust_remote_code: bool = DEFAULT_PLACE_TRUST_REMOTE_CODE,
        runtime: str = "hf_transformers",
    ):
        self.model_id = model_id
        self.model_name = model_name
        self.pretrained = str(pretrained or "")
        self.model_version = model_version
        self.weight = float(weight)
        self.embedding_base = EMBEDDING_BASE_PLACE
        self.runtime = str(runtime or "hf_transformers")
        self.trust_remote_code = bool(trust_remote_code)
        self._torch = None
        self._processor = None
        self._model = None
        self._clip_preprocess = None
        self._device = "cpu"
        self._hf_token = DEFAULT_HF_TOKEN

    def _ensure_loaded(self):
        if self._model is not None and self._processor is not None:
            return
        if self.runtime == "open_clip":
            torch, open_clip = _import_clip_runtime()
            self._torch = torch
            if torch.cuda.is_available():
                self._device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self._device = "mps"
            else:
                self._device = "cpu"
            if not self.pretrained:
                raise RuntimeError(
                    "Place runtime=open_clip requires pretrained checkpoint name."
                )
            model, _, preprocess = open_clip.create_model_and_transforms(
                self.model_name, pretrained=self.pretrained
            )
            model.eval()
            model.to(self._device)
            self._model = model
            self._clip_preprocess = preprocess
            self._processor = "open_clip"
            return
        torch, AutoImageProcessor, AutoModel = _import_hf_runtime()
        self._torch = torch
        if torch.cuda.is_available():
            self._device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self._device = "mps"
        else:
            self._device = "cpu"
        self._processor = AutoImageProcessor.from_pretrained(
            self.model_name,
            trust_remote_code=self.trust_remote_code,
            token=self._hf_token,
        )
        self._model = AutoModel.from_pretrained(
            self.model_name,
            trust_remote_code=self.trust_remote_code,
            token=self._hf_token,
        )
        self._model.eval()
        self._model.to(self._device)

    def _extract_features(self, outputs: Any):
        if self.runtime == "open_clip":
            return outputs
        if getattr(outputs, "pooler_output", None) is not None:
            feats = outputs.pooler_output
        elif getattr(outputs, "last_hidden_state", None) is not None:
            feats = outputs.last_hidden_state.mean(dim=1)
        elif isinstance(outputs, (tuple, list)) and outputs:
            feats = outputs[0]
            if getattr(feats, "dim", lambda: 0)() == 3:
                feats = feats.mean(dim=1)
        else:
            raise RuntimeError("Unable to extract embeddings from place model output")
        if feats.dim() == 1:
            feats = feats.unsqueeze(0)
        return feats

    @property
    def embedding_dim(self) -> int:
        self._ensure_loaded()
        hidden_size = getattr(getattr(self._model, "config", None), "hidden_size", None)
        if hidden_size is not None:
            return int(hidden_size)
        image = Image.new("RGB", (32, 32))
        out = io.BytesIO()
        image.save(out, format="JPEG", quality=90)
        probe = self.encode_image_bytes_batch([out.getvalue()])
        if not probe or not probe[0]:
            raise RuntimeError("Unable to infer place embedding dimension")
        return len(probe[0])

    def encode_image_bytes_batch(self, image_bytes_batch: List[bytes]) -> List[List[float]]:
        self._ensure_loaded()
        if not image_bytes_batch:
            return []
        if self.runtime == "open_clip":
            tensors = []
            for image_bytes in image_bytes_batch:
                image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                tensors.append(self._clip_preprocess(image))
            batch = self._torch.stack(tensors, dim=0).to(self._device)
            with self._torch.no_grad():
                features = self._model.encode_image(batch)
                features = features / features.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        else:
            images = [
                Image.open(io.BytesIO(image_bytes)).convert("RGB")
                for image_bytes in image_bytes_batch
            ]
            inputs = self._processor(images=images, return_tensors="pt")
            for key, value in list(inputs.items()):
                if hasattr(value, "to"):
                    inputs[key] = value.to(self._device)
            with self._torch.no_grad():
                outputs = self._model(**inputs)
                features = self._extract_features(outputs)
                features = features / features.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        return features.detach().cpu().float().tolist()

    def encode_image_bytes(self, image_bytes: bytes) -> List[float]:
        batch = self.encode_image_bytes_batch([image_bytes])
        return batch[0] if batch else []


def _build_embedder(cfg: RetrievalModelConfig):
    if cfg.runtime == "hf_transformers":
        return PlaceEmbedder(
            model_id=cfg.model_id,
            model_name=cfg.model_name,
            pretrained=cfg.pretrained,
            model_version=cfg.model_version,
            weight=cfg.weight,
            trust_remote_code=cfg.trust_remote_code,
            runtime=cfg.runtime,
        )
    if cfg.model_id == "place":
        return PlaceEmbedder(
            model_id=cfg.model_id,
            model_name=cfg.model_name,
            pretrained=cfg.pretrained,
            model_version=cfg.model_version,
            weight=cfg.weight,
            trust_remote_code=cfg.trust_remote_code,
            runtime=cfg.runtime,
        )
    return ClipEmbedder(
        model_id=cfg.model_id,
        model_name=cfg.model_name,
        pretrained=cfg.pretrained,
        model_version=cfg.model_version,
        weight=cfg.weight,
    )


@lru_cache(maxsize=1)
def get_retrieval_embedders() -> Tuple[Any, ...]:
    return tuple(_build_embedder(cfg) for cfg in _build_retrieval_model_configs())


def select_retrieval_embedders(
    embedding_base: str, allow_fallback: bool = True
) -> Tuple[Any, ...]:
    base = str(embedding_base or EMBEDDING_BASE_CLIP).strip().lower()
    selected = tuple(
        embedder
        for embedder in get_retrieval_embedders()
        if str(getattr(embedder, "embedding_base", EMBEDDING_BASE_CLIP)).lower() == base
    )
    if selected:
        return selected
    if allow_fallback and base != EMBEDDING_BASE_CLIP:
        return select_retrieval_embedders(
            EMBEDDING_BASE_CLIP, allow_fallback=allow_fallback
        )
    return tuple()


@lru_cache(maxsize=1)
def get_clip_embedder():
    selected = select_retrieval_embedders(EMBEDDING_BASE_CLIP)
    if not selected:
        raise RuntimeError("CLIP retrieval embedder is not configured")
    return selected[0]


def get_retrieval_embedder_keys() -> List[Tuple[str, str, str, float]]:
    keys: List[Tuple[str, str, str, float]] = []
    for embedder in get_retrieval_embedders():
        keys.append(
            (
                embedder.model_id,
                embedder.model_name,
                embedder.model_version,
                float(embedder.weight),
            )
        )
    return keys


def encode_image_for_all_models(
    image_bytes: bytes,
    embedders: Optional[Sequence[Any]] = None,
) -> List[Tuple[Any, List[float]]]:
    results: List[Tuple[Any, List[float]]] = []
    for embedder in list(embedders or get_retrieval_embedders()):
        try:
            results.append((embedder, embedder.encode_image_bytes(image_bytes)))
        except Exception as exc:
            log.warning(
                "Skipping retrieval model due to encode failure model_id=%s model=%s version=%s error=%s",
                embedder.model_id,
                embedder.model_name,
                embedder.model_version,
                exc,
            )
    return results


def encode_image_batch_for_all_models(
    image_bytes_batch: Sequence[bytes],
    embedders: Optional[Sequence[Any]] = None,
) -> List[Tuple[Any, List[List[float]]]]:
    results: List[Tuple[Any, List[List[float]]]] = []
    batch = list(image_bytes_batch)
    for embedder in list(embedders or get_retrieval_embedders()):
        try:
            results.append((embedder, embedder.encode_image_bytes_batch(batch)))
        except Exception as exc:
            log.warning(
                "Skipping retrieval model due to batch encode failure model_id=%s model=%s version=%s error=%s",
                embedder.model_id,
                embedder.model_name,
                embedder.model_version,
                exc,
            )
    return results
