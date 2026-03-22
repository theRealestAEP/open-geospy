import logging
import os
from typing import List, Optional, Tuple

from backend.app.clip_embeddings import (
    EMBEDDING_BASE_CLIP,
    encode_image_batch_for_all_models,
    get_retrieval_embedders,
)
from backend.app.vector_store import build_vector_store

DEFAULT_EMBED_ON_INGEST = os.getenv("GEOSPY_EMBED_ON_INGEST", "1").strip().lower() not in {
    "0",
    "false",
    "no",
    "off",
}
DEFAULT_EMBED_BATCH_SIZE = max(1, int(os.getenv("GEOSPY_EMBED_BATCH_SIZE", "128")))


class CaptureEmbeddingIngestor:
    def __init__(
        self,
        db,
        *,
        enabled: bool = DEFAULT_EMBED_ON_INGEST,
        batch_size: int = DEFAULT_EMBED_BATCH_SIZE,
        logger: Optional[logging.Logger] = None,
        vector_store=None,
    ):
        self.db = db
        self.log = logger or logging.getLogger(__name__)
        self.vector_store = vector_store or build_vector_store(db)
        self.batch_size = max(1, int(batch_size))
        self.enabled = False
        self.saved_embeddings = 0
        self.embed_errors = 0
        self.embedders: List = []
        self.pending_capture_images: List[Tuple[int, bytes]] = []

        if not enabled:
            return
        if not self.vector_store.supports_writes():
            self.log.warning(
                "ingest embedding disabled; vector backend=%s does not support writes",
                getattr(self.vector_store, "backend_name", "unknown"),
            )
            return

        ready_check = getattr(self.vector_store, "is_write_ready", None)
        is_write_ready = (
            bool(ready_check()) if callable(ready_check) else bool(self.vector_store.is_vector_ready())
        )
        if not is_write_ready:
            self.log.warning(
                "ingest embedding disabled; vector backend=%s is not write-ready",
                getattr(self.vector_store, "backend_name", "unknown"),
            )
            return

        try:
            self.embedders = list(get_retrieval_embedders())
        except Exception as exc:
            self.log.warning("ingest embedding disabled; embedder unavailable: %s", exc)
            return
        if not self.embedders:
            self.log.warning("ingest embedding disabled; no retrieval embedders configured")
            return

        self.enabled = True
        self.log.info(
            "ingest embedding enabled backend=%s models=%s batch_size=%s",
            getattr(self.vector_store, "backend_name", "unknown"),
            [
                f"{embedder.model_name}:{embedder.model_version}"
                for embedder in self.embedders
            ],
            self.batch_size,
        )

    def add_capture(self, capture_id: int, image_bytes: bytes) -> None:
        if not self.enabled or not image_bytes:
            return
        self.pending_capture_images.append((int(capture_id), image_bytes))
        if len(self.pending_capture_images) >= self.batch_size:
            self.flush()

    def flush(self) -> None:
        if not self.enabled or not self.pending_capture_images:
            return

        pending_batch = list(self.pending_capture_images)
        self.pending_capture_images = []
        try:
            model_vectors = encode_image_batch_for_all_models(
                [image_bytes for _, image_bytes in pending_batch],
                embedders=self.embedders,
            )
            if not model_vectors:
                raise RuntimeError("No retrieval models encoded successfully")
            for embedder, vectors in model_vectors:
                if len(vectors) != len(pending_batch):
                    raise RuntimeError(
                        f"batch embedding size mismatch for {embedder.model_id} "
                        f"expected={len(pending_batch)} got={len(vectors)}"
                    )
                self.saved_embeddings += int(
                    self.vector_store.upsert_capture_embeddings_batch(
                        [
                            (capture_id, vector)
                            for (capture_id, _), vector in zip(pending_batch, vectors)
                        ],
                        embedder.model_name,
                        embedder.model_version,
                        embedding_base=str(
                            getattr(embedder, "embedding_base", EMBEDDING_BASE_CLIP)
                        ),
                    )
                )
        except Exception as exc:
            self.log.warning(
                "ingest embedding batch failed size=%s error=%s; retrying singles",
                len(pending_batch),
                exc,
            )
            for capture_id, image_bytes in pending_batch:
                try:
                    encoded_any = False
                    for embedder in self.embedders:
                        vector = embedder.encode_image_bytes(image_bytes)
                        self.vector_store.upsert_capture_embedding(
                            capture_id,
                            embedder.model_name,
                            embedder.model_version,
                            vector,
                            embedding_base=str(
                                getattr(embedder, "embedding_base", EMBEDDING_BASE_CLIP)
                            ),
                        )
                        self.saved_embeddings += 1
                        encoded_any = True
                    if not encoded_any:
                        self.embed_errors += 1
                except Exception as single_exc:
                    self.embed_errors += 1
                    self.log.warning(
                        "ingest embedding failed capture_id=%s error=%s",
                        capture_id,
                        single_exc,
                    )

    def close(self) -> None:
        self.flush()
