import logging
import os
from typing import Dict, List, Optional, Sequence, Tuple

from backend.app.clip_embeddings import EMBEDDING_BASE_CLIP

log = logging.getLogger(__name__)

VECTOR_BACKEND_POSTGRES = "postgres"
VECTOR_BACKEND_LANCEDB = "lancedb"


def normalize_vector_backend(value: Optional[str]) -> str:
    raw = str(value or VECTOR_BACKEND_LANCEDB).strip().lower()
    if raw in {"lance", VECTOR_BACKEND_LANCEDB}:
        return VECTOR_BACKEND_LANCEDB
    return VECTOR_BACKEND_POSTGRES


def get_configured_vector_backend() -> str:
    return normalize_vector_backend(os.getenv("GEOSPY_VECTOR_BACKEND", VECTOR_BACKEND_LANCEDB))


def _quote_sql_literal(value: str) -> str:
    escaped = str(value).replace("'", "''")
    return f"'{escaped}'"


class PostgresVectorStore:
    backend_name = VECTOR_BACKEND_POSTGRES

    def __init__(self, db):
        self.db = db

    def supports_writes(self) -> bool:
        return True

    def supports_missing_embedding_backfill(self) -> bool:
        return True

    def is_write_ready(self) -> bool:
        return self.is_vector_ready()

    def is_vector_ready(self) -> bool:
        return bool(self.db.is_vector_ready())

    def get_capture_embedding_stats(
        self, model_name: str, model_version: str, embedding_base: str = EMBEDDING_BASE_CLIP
    ) -> dict:
        return self.db.get_capture_embedding_stats(
            model_name, model_version, embedding_base=embedding_base
        )

    def list_captures_missing_any_embeddings(
        self,
        models: Sequence[Tuple[str, str]],
        limit: int = 64,
        after_capture_id: int = 0,
        embedding_base: str = EMBEDDING_BASE_CLIP,
    ) -> List[dict]:
        return self.db.list_captures_missing_any_embeddings(
            models,
            limit=limit,
            after_capture_id=after_capture_id,
            embedding_base=embedding_base,
        )

    def upsert_capture_embedding(
        self,
        capture_id: int,
        model_name: str,
        model_version: str,
        embedding: Sequence[float],
        embedding_base: str = EMBEDDING_BASE_CLIP,
    ):
        return self.db.upsert_capture_embedding(
            capture_id,
            model_name,
            model_version,
            embedding,
            embedding_base=embedding_base,
        )

    def upsert_capture_embeddings_batch(
        self,
        capture_vectors: Sequence[Tuple[int, Sequence[float]]],
        model_name: str,
        model_version: str,
        embedding_base: str = EMBEDDING_BASE_CLIP,
    ) -> int:
        return self.db.upsert_capture_embeddings_batch(
            capture_vectors,
            model_name,
            model_version,
            embedding_base=embedding_base,
        )

    def search_captures_by_embedding(
        self,
        embedding: Sequence[float],
        model_name: str,
        model_version: str,
        top_k: int = 12,
        min_similarity: Optional[float] = None,
        max_top_k: int = 200,
        trace_id: Optional[str] = None,
        ivfflat_probes: Optional[int] = None,
        embedding_base: str = EMBEDDING_BASE_CLIP,
    ) -> List[dict]:
        return self.db.search_captures_by_embedding(
            embedding=embedding,
            model_name=model_name,
            model_version=model_version,
            top_k=top_k,
            min_similarity=min_similarity,
            max_top_k=max_top_k,
            trace_id=trace_id,
            ivfflat_probes=ivfflat_probes,
            embedding_base=embedding_base,
        )


class LanceVectorStore:
    backend_name = VECTOR_BACKEND_LANCEDB
    _connection_cache: Dict[str, object] = {}

    def __init__(
        self,
        db,
        *,
        uri: str,
        table_name: str,
        vector_column_name: str = "embedding",
        embedding_base_column: str = "",
    ):
        self.db = db
        self.uri = str(uri or ".lancedb")
        self.table_name = str(table_name or "capture_embeddings")
        self.vector_column_name = str(vector_column_name or "embedding")
        self.embedding_base_column = str(embedding_base_column or "").strip()

    def supports_writes(self) -> bool:
        return True

    def supports_missing_embedding_backfill(self) -> bool:
        return False

    def is_write_ready(self) -> bool:
        try:
            self._connect()
            return True
        except Exception:
            return False

    def _connect(self):
        cached = self._connection_cache.get(self.uri)
        if cached is not None:
            return cached
        try:
            import lancedb
        except ImportError as exc:
            raise RuntimeError(
                "LanceDB backend selected but dependency is missing. Install with: pip install lancedb"
            ) from exc
        kwargs = {}
        if self.uri.startswith("db://"):
            api_key = os.getenv("LANCEDB_API_KEY", "").strip()
            if api_key:
                kwargs["api_key"] = api_key
            region = os.getenv("LANCEDB_REGION", "").strip()
            if region:
                kwargs["region"] = region
            host_override = os.getenv("LANCEDB_HOST_OVERRIDE", "").strip()
            if host_override:
                kwargs["host_override"] = host_override
        conn = lancedb.connect(self.uri, **kwargs)
        self._connection_cache[self.uri] = conn
        return conn

    def _open_table_or_none(self):
        conn = self._connect()
        try:
            names = conn.table_names()
        except Exception as exc:
            log.warning("Failed to list LanceDB tables uri=%s error=%s", self.uri, exc)
            return None
        if self.table_name not in set(names):
            return None
        return conn.open_table(self.table_name)

    def _require_table(self):
        table = self._open_table_or_none()
        if table is None:
            raise RuntimeError(
                f"LanceDB table '{self.table_name}' not found at '{self.uri}'. "
                "Run the pgvector -> LanceDB sync script first."
            )
        return table

    def _table_uses_embedding_base_column(self, table) -> bool:
        if not self.embedding_base_column:
            return False
        try:
            schema = table.schema
            names = set(getattr(schema, "names", []) or [])
        except Exception:
            names = set()
        return self.embedding_base_column in names

    def _build_lance_payload(
        self,
        capture_vectors: Sequence[Tuple[int, Sequence[float]]],
        *,
        model_name: str,
        model_version: str,
        embedding_base: str,
        include_embedding_base: bool,
    ) -> List[dict]:
        payload: List[dict] = []
        for capture_id, embedding in capture_vectors:
            row = {
                "capture_id": int(capture_id),
                "model_name": str(model_name),
                "model_version": str(model_version),
                self.vector_column_name: [float(v) for v in embedding],
            }
            if include_embedding_base and self.embedding_base_column:
                row[self.embedding_base_column] = str(embedding_base)
            payload.append(row)
        return payload

    def _create_table_for_payload(self, payload: List[dict], *, include_embedding_base: bool):
        conn = self._connect()
        try:
            import pyarrow as pa
        except ImportError:
            return conn.create_table(self.table_name, data=payload)

        first_vector = list(payload[0].get(self.vector_column_name) or [])
        embedding_dim = len(first_vector)
        if embedding_dim <= 0:
            raise RuntimeError("Cannot create LanceDB table with empty embedding vectors")

        fields = [
            pa.field("capture_id", pa.int64(), nullable=False),
            pa.field("model_name", pa.string(), nullable=False),
            pa.field("model_version", pa.string(), nullable=False),
        ]
        if include_embedding_base and self.embedding_base_column:
            fields.append(pa.field(self.embedding_base_column, pa.string(), nullable=False))
        fields.append(
            pa.field(
                self.vector_column_name,
                pa.list_(pa.float32(), embedding_dim),
                nullable=False,
            )
        )
        return conn.create_table(
            self.table_name,
            data=payload,
            schema=pa.schema(fields),
        )

    def _delete_existing_rows(self, table, payload: Sequence[dict], *, include_embedding_base: bool):
        delete_terms: List[str] = []
        for row in payload:
            clauses = [
                f"capture_id = {int(row['capture_id'])}",
                f"model_name = {_quote_sql_literal(str(row['model_name']))}",
                f"model_version = {_quote_sql_literal(str(row['model_version']))}",
            ]
            if include_embedding_base and self.embedding_base_column:
                clauses.append(
                    f"{self.embedding_base_column} = "
                    f"{_quote_sql_literal(str(row[self.embedding_base_column]))}"
                )
            delete_terms.append(f"({' AND '.join(clauses)})")
            if len(delete_terms) >= 128:
                table.delete(" OR ".join(delete_terms))
                delete_terms = []
        if delete_terms:
            table.delete(" OR ".join(delete_terms))

    def _model_filter(
        self, model_name: str, model_version: str, embedding_base: str
    ) -> str:
        parts = [
            f"model_name = {_quote_sql_literal(model_name)}",
            f"model_version = {_quote_sql_literal(model_version)}",
        ]
        if self.embedding_base_column:
            parts.append(
                f"{self.embedding_base_column} = {_quote_sql_literal(embedding_base)}"
            )
        return " AND ".join(parts)

    def is_vector_ready(self) -> bool:
        try:
            return self._open_table_or_none() is not None
        except Exception:
            return False

    def get_capture_embedding_stats(
        self, model_name: str, model_version: str, embedding_base: str = EMBEDDING_BASE_CLIP
    ) -> dict:
        total_captures = int(self.db.get_total_capture_count())
        table = self._open_table_or_none()
        if table is None:
            return {
                "vector_enabled": False,
                "embedding_base": embedding_base,
                "model_name": model_name,
                "model_version": model_version,
                "total_captures": total_captures,
                "embedded_captures": 0,
                "pending_captures": total_captures,
            }
        filter_expr = self._model_filter(model_name, model_version, embedding_base)
        try:
            embedded_captures = int(table.count_rows(filter_expr))
        except Exception as exc:
            log.warning("LanceDB count_rows failed filter=%s error=%s", filter_expr, exc)
            embedded_captures = 0
        return {
            "vector_enabled": True,
            "embedding_base": embedding_base,
            "model_name": model_name,
            "model_version": model_version,
            "total_captures": total_captures,
            "embedded_captures": embedded_captures,
            "pending_captures": max(0, total_captures - embedded_captures),
        }

    def list_captures_missing_any_embeddings(
        self,
        models: Sequence[Tuple[str, str]],
        limit: int = 64,
        after_capture_id: int = 0,
        embedding_base: str = EMBEDDING_BASE_CLIP,
    ) -> List[dict]:
        raise RuntimeError(
            "LanceDB backend does not yet support missing-embedding backfill queries"
        )

    def upsert_capture_embedding(
        self,
        capture_id: int,
        model_name: str,
        model_version: str,
        embedding: Sequence[float],
        embedding_base: str = EMBEDDING_BASE_CLIP,
    ):
        return self.upsert_capture_embeddings_batch(
            [(capture_id, embedding)],
            model_name,
            model_version,
            embedding_base=embedding_base,
        )

    def upsert_capture_embeddings_batch(
        self,
        capture_vectors: Sequence[Tuple[int, Sequence[float]]],
        model_name: str,
        model_version: str,
        embedding_base: str = EMBEDDING_BASE_CLIP,
    ) -> int:
        rows = list(capture_vectors)
        if not rows:
            return 0

        table = self._open_table_or_none()
        include_embedding_base = bool(self.embedding_base_column)
        if table is not None:
            include_embedding_base = self._table_uses_embedding_base_column(table)
            if self.embedding_base_column and not include_embedding_base:
                log.warning(
                    "LanceDB table=%s is missing embedding base column=%s; writing without it",
                    self.table_name,
                    self.embedding_base_column,
                )

        payload = self._build_lance_payload(
            rows,
            model_name=model_name,
            model_version=model_version,
            embedding_base=embedding_base,
            include_embedding_base=include_embedding_base,
        )
        if table is None:
            self._create_table_for_payload(
                payload,
                include_embedding_base=include_embedding_base,
            )
            return len(payload)

        self._delete_existing_rows(
            table,
            payload,
            include_embedding_base=include_embedding_base,
        )
        table.add(payload)
        return len(payload)

    def search_captures_by_embedding(
        self,
        embedding: Sequence[float],
        model_name: str,
        model_version: str,
        top_k: int = 12,
        min_similarity: Optional[float] = None,
        max_top_k: int = 200,
        trace_id: Optional[str] = None,
        ivfflat_probes: Optional[int] = None,
        embedding_base: str = EMBEDDING_BASE_CLIP,
    ) -> List[dict]:
        table = self._require_table()
        limit = max(1, min(int(max_top_k), int(top_k)))
        filter_expr = self._model_filter(model_name, model_version, embedding_base)
        query = table.search(
            [float(v) for v in embedding], vector_column_name=self.vector_column_name
        ).metric("cosine")
        if ivfflat_probes is not None and hasattr(query, "nprobes"):
            try:
                query = query.nprobes(max(1, int(ivfflat_probes)))
            except Exception:
                pass
        query = query.where(filter_expr, prefilter=True).select(["capture_id"]).limit(limit)
        rows = query.to_list()

        ranked_hits: List[Tuple[int, float]] = []
        similarity_floor = float(min_similarity) if min_similarity is not None else None
        for row in rows:
            capture_id = row.get("capture_id")
            if capture_id is None:
                continue
            distance = float(row.get("_distance", 1.0))
            similarity = max(-1.0, min(1.0, 1.0 - distance))
            if similarity_floor is not None and similarity < similarity_floor:
                continue
            ranked_hits.append((int(capture_id), similarity))

        if not ranked_hits:
            return []

        metadata = self.db.get_capture_metadata_for_capture_ids(
            [capture_id for capture_id, _ in ranked_hits]
        )
        result: List[dict] = []
        for capture_id, similarity in ranked_hits:
            row = metadata.get(capture_id)
            if not row:
                continue
            payload = dict(row)
            payload["similarity"] = similarity
            result.append(payload)
        if trace_id:
            log.info(
                "retrieval_lance trace_id=%s top_k=%s returned=%s model=%s version=%s",
                trace_id,
                limit,
                len(result),
                model_name,
                model_version,
            )
        return result


def build_vector_store(db):
    backend = get_configured_vector_backend()
    if backend == VECTOR_BACKEND_LANCEDB:
        return LanceVectorStore(
            db,
            uri=os.getenv("GEOSPY_LANCEDB_URI", ".lancedb"),
            table_name=os.getenv("GEOSPY_LANCEDB_TABLE", "capture_embeddings"),
            vector_column_name=os.getenv("GEOSPY_LANCEDB_VECTOR_COLUMN", "embedding"),
            embedding_base_column=os.getenv("GEOSPY_LANCEDB_EMBEDDING_BASE_COLUMN", ""),
        )
    return PostgresVectorStore(db)
