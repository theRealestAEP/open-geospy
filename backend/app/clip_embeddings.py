import io
import os
from dataclasses import dataclass
from functools import lru_cache
from typing import List, Sequence, Tuple

from PIL import Image

import logging

log = logging.getLogger(__name__)

DEFAULT_CLIP_MODEL = os.getenv("GEOSPY_CLIP_MODEL", "ViT-B-32")
DEFAULT_CLIP_PRETRAINED = os.getenv("GEOSPY_CLIP_PRETRAINED", "laion2b_s34b_b79k")
DEFAULT_CLIP_VERSION = os.getenv("GEOSPY_CLIP_VERSION", "open_clip")
DEFAULT_PLACE_MODEL = os.getenv("GEOSPY_PLACE_MODEL", "ViT-B-16")
DEFAULT_PLACE_PRETRAINED = os.getenv(
    "GEOSPY_PLACE_PRETRAINED", "laion2b_s34b_b88k"
)
DEFAULT_PLACE_VERSION = os.getenv("GEOSPY_PLACE_VERSION", "open_clip_place")


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
    weight: float = 1.0
    enabled: bool = True


def _build_retrieval_model_configs() -> List[RetrievalModelConfig]:
    primary_weight = max(
        0.0, float(os.getenv("GEOSPY_RETRIEVAL_PRIMARY_WEIGHT", "1.0"))
    )
    place_enabled = _is_enabled(
        os.getenv("GEOSPY_PLACE_MODEL_ENABLED", "1"), default=True
    )
    place_weight = max(0.0, float(os.getenv("GEOSPY_RETRIEVAL_PLACE_WEIGHT", "0.9")))

    configs: List[RetrievalModelConfig] = [
        RetrievalModelConfig(
            model_id="clip",
            model_name=DEFAULT_CLIP_MODEL,
            pretrained=DEFAULT_CLIP_PRETRAINED,
            model_version=DEFAULT_CLIP_VERSION,
            weight=primary_weight,
            enabled=True,
        )
    ]
    if place_enabled:
        configs.append(
            RetrievalModelConfig(
                model_id="place",
                model_name=DEFAULT_PLACE_MODEL,
                pretrained=DEFAULT_PLACE_PRETRAINED,
                model_version=DEFAULT_PLACE_VERSION,
                weight=place_weight,
                enabled=True,
            )
        )
    deduped: List[RetrievalModelConfig] = []
    seen = set()
    for cfg in configs:
        key = (cfg.model_id, cfg.model_name, cfg.pretrained, cfg.model_version)
        if key in seen or cfg.weight <= 0:
            continue
        seen.add(key)
        deduped.append(cfg)
    return deduped


def _import_runtime():
    try:
        import open_clip
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "CLIP dependencies are missing. Install open-clip-torch and torch."
        ) from exc
    return torch, open_clip


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
        self._torch = None
        self._tokenizer = None
        self._model = None
        self._preprocess = None
        self._device = "cpu"
        self._query_adapter = None
        self._query_adapter_loaded = False

    def _ensure_loaded(self):
        if self._model is not None and self._preprocess is not None:
            return
        torch, open_clip = _import_runtime()
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
        self._tokenizer = open_clip.get_tokenizer(self.model_name)
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
            log.info(
                "Loaded query adapter model_id=%s path=%s",
                self.model_id,
                adapter_path,
            )
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

    def encode_text(self, query: str) -> List[float]:
        self._ensure_loaded()
        tokens = self._tokenizer([query]).to(self._device)
        with self._torch.no_grad():
            features = self._model.encode_text(tokens)
            features = features / features.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        return features[0].detach().cpu().float().tolist()


@lru_cache(maxsize=1)
def get_clip_embedder() -> ClipEmbedder:
    return get_retrieval_embedders()[0]


@lru_cache(maxsize=1)
def get_retrieval_embedders() -> Tuple[ClipEmbedder, ...]:
    configs = _build_retrieval_model_configs()
    embedders: List[ClipEmbedder] = []
    for cfg in configs:
        embedders.append(
            ClipEmbedder(
                model_id=cfg.model_id,
                model_name=cfg.model_name,
                pretrained=cfg.pretrained,
                model_version=cfg.model_version,
                weight=cfg.weight,
            )
        )
    return tuple(embedders)


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


def encode_image_for_all_models(image_bytes: bytes) -> List[Tuple[ClipEmbedder, List[float]]]:
    results: List[Tuple[ClipEmbedder, List[float]]] = []
    for embedder in get_retrieval_embedders():
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
) -> List[Tuple[ClipEmbedder, List[List[float]]]]:
    results: List[Tuple[ClipEmbedder, List[List[float]]]] = []
    batch = list(image_bytes_batch)
    for embedder in get_retrieval_embedders():
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
