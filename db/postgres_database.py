"""PostgreSQL database layer for panorama metadata, deduplication, and task claiming."""

import logging
import math
import os
from dataclasses import dataclass
from datetime import date, datetime
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import psycopg
from psycopg.errors import UniqueViolation
from psycopg.rows import dict_row

log = logging.getLogger(__name__)


DEFAULT_DATABASE_URL = os.getenv(
    "DATABASE_URL", "postgresql://geospy:geospy@127.0.0.1:5432/geospy"
)
DEFAULT_EMBEDDING_DIM = int(os.getenv("GEOSPY_EMBEDDING_DIM", "512"))
DEFAULT_RETRIEVAL_IVFFLAT_PROBES = max(
    1, int(os.getenv("GEOSPY_RETRIEVAL_IVFFLAT_PROBES", "120"))
)
EMBEDDING_BASE_CLIP = "clip"
EMBEDDING_BASE_PLACE = "place"
CAPTURE_EMBEDDINGS_TABLE = "capture_embeddings"


@dataclass
class Panorama:
    id: Optional[int]
    lat: float
    lon: float
    pano_id: Optional[str]
    heading: float
    pitch: float
    timestamp: str
    source_url: str
    city: str = ""
    notes: str = ""


@dataclass
class Capture:
    id: Optional[int]
    panorama_id: int
    heading: float
    filepath: str
    width: int
    height: int
    pitch: float = 75.0
    capture_profile: str = "base"
    capture_kind: str = "scan"
    is_black_frame: int = 0
    quality_reason: str = ""
    brightness_mean: Optional[float] = None
    flagged_at: Optional[str] = None


class Database:
    def __init__(self, db_url: str = DEFAULT_DATABASE_URL):
        self.db_url = db_url
        self.embedding_dim = DEFAULT_EMBEDDING_DIM
        self.vector_enabled = False
        self.conn = psycopg.connect(db_url, row_factory=dict_row)
        self._configure_connection()
        self._init_tables()

    def _configure_connection(self):
        self.conn.execute("SET statement_timeout = 0")

    def _init_tables(self):
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS panoramas (
                id BIGSERIAL PRIMARY KEY,
                lat DOUBLE PRECISION NOT NULL,
                lon DOUBLE PRECISION NOT NULL,
                pano_id TEXT,
                heading DOUBLE PRECISION NOT NULL DEFAULT 0,
                pitch DOUBLE PRECISION NOT NULL DEFAULT 0,
                timestamp TIMESTAMPTZ NOT NULL,
                source_url TEXT,
                city TEXT NOT NULL DEFAULT '',
                notes TEXT NOT NULL DEFAULT '',
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
            """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS captures (
                id BIGSERIAL PRIMARY KEY,
                panorama_id BIGINT NOT NULL REFERENCES panoramas(id) ON DELETE CASCADE,
                heading DOUBLE PRECISION NOT NULL,
                pitch DOUBLE PRECISION NOT NULL DEFAULT 75,
                filepath TEXT NOT NULL,
                width INTEGER,
                height INTEGER,
                capture_profile TEXT NOT NULL DEFAULT 'base',
                capture_kind TEXT NOT NULL DEFAULT 'scan',
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
            """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS seed_tasks (
                id BIGSERIAL PRIMARY KEY,
                lat DOUBLE PRECISION NOT NULL,
                lon DOUBLE PRECISION NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending',
                claimed_by TEXT,
                claimed_at TIMESTAMPTZ,
                attempts INTEGER NOT NULL DEFAULT 0,
                last_error TEXT NOT NULL DEFAULT '',
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                UNIQUE(lat, lon)
            )
            """
        )

        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_panoramas_lat_lon ON panoramas(lat, lon)"
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_panoramas_pano_id ON panoramas(pano_id)"
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_captures_panorama_id ON captures(panorama_id)"
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_seed_tasks_status ON seed_tasks(status)"
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_seed_tasks_claimed_at ON seed_tasks(claimed_at)"
        )

        try:
            self.conn.execute(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS idx_panoramas_pano_id_unique
                ON panoramas(pano_id)
                WHERE pano_id IS NOT NULL
                """
            )
        except Exception:
            log.warning("Could not create unique pano_id index due to existing duplicates")

        self._ensure_column("captures", "is_black_frame", "INTEGER NOT NULL DEFAULT 0")
        self._ensure_column("captures", "quality_reason", "TEXT NOT NULL DEFAULT ''")
        self._ensure_column("captures", "brightness_mean", "DOUBLE PRECISION")
        self._ensure_column("captures", "flagged_at", "TIMESTAMPTZ")
        self._ensure_column("captures", "pitch", "DOUBLE PRECISION NOT NULL DEFAULT 75")
        self._ensure_column(
            "captures", "capture_profile", "TEXT NOT NULL DEFAULT 'base'"
        )
        self._ensure_column("captures", "capture_kind", "TEXT NOT NULL DEFAULT 'scan'")
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_captures_is_black ON captures(is_black_frame)"
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_captures_profile_kind ON captures(capture_profile, capture_kind)"
        )
        self._init_vector_schema()

        self.conn.commit()

    def _ensure_column(self, table: str, column: str, ddl_fragment: str):
        row = self.conn.execute(
            """
            SELECT 1
            FROM information_schema.columns
            WHERE table_schema = 'public'
              AND table_name = %s
              AND column_name = %s
            LIMIT 1
            """,
            (table, column),
        ).fetchone()
        if row:
            return
        self.conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {ddl_fragment}")

    def _normalize_embedding_base(self, embedding_base: Optional[str]) -> str:
        raw = str(embedding_base or EMBEDDING_BASE_CLIP).strip().lower()
        if raw in {EMBEDDING_BASE_CLIP, EMBEDDING_BASE_PLACE}:
            return raw
        return EMBEDDING_BASE_CLIP

    def _embedding_table_for_base(self, embedding_base: Optional[str]) -> str:
        self._normalize_embedding_base(embedding_base)
        return CAPTURE_EMBEDDINGS_TABLE

    def _init_vector_schema(self):
        self.conn.execute("SAVEPOINT vector_init")
        try:
            self.conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            self.conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {CAPTURE_EMBEDDINGS_TABLE} (
                    capture_id BIGINT NOT NULL REFERENCES captures(id) ON DELETE CASCADE,
                    model_name TEXT NOT NULL,
                    model_version TEXT NOT NULL DEFAULT '',
                    embedding vector({self.embedding_dim}) NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    PRIMARY KEY (capture_id, model_name, model_version)
                )
                """
            )
            self._migrate_capture_embeddings_primary_key(CAPTURE_EMBEDDINGS_TABLE)
            self.conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_capture_embeddings_model
                ON capture_embeddings(model_name, model_version)
                """
            )
            self.conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_capture_embeddings_embedding_ivfflat_global
                ON capture_embeddings
                USING ivfflat (embedding vector_cosine_ops)
                WITH (lists = 100)
                """
            )
            self._ensure_model_specific_vector_indexes()
            self.conn.execute("RELEASE SAVEPOINT vector_init")
            self.vector_enabled = True
        except Exception as exc:
            self.conn.execute("ROLLBACK TO SAVEPOINT vector_init")
            self.conn.execute("RELEASE SAVEPOINT vector_init")
            self.vector_enabled = False
            log.warning("Vector schema init failed; retrieval disabled: %s", exc)

    def _migrate_capture_embeddings_primary_key(self, table_name: str):
        pk_info = self.conn.execute(
            """
            SELECT
                c.conname AS constraint_name,
                array_agg(a.attname ORDER BY x.ordinality) AS columns
            FROM pg_constraint c
            JOIN pg_class t ON t.oid = c.conrelid
            JOIN LATERAL unnest(c.conkey) WITH ORDINALITY AS x(attnum, ordinality) ON TRUE
            JOIN pg_attribute a ON a.attrelid = t.oid AND a.attnum = x.attnum
            WHERE c.contype = 'p'
              AND t.relname = %s
            GROUP BY c.conname
            LIMIT 1
            """,
            (str(table_name),),
        ).fetchone()
        if not pk_info:
            self.conn.execute(
                f"""
                ALTER TABLE {table_name}
                ADD CONSTRAINT {table_name}_pkey
                PRIMARY KEY (capture_id, model_name, model_version)
                """
            )
            return
        existing_cols = [str(c) for c in (pk_info.get("columns") or [])]
        if existing_cols == ["capture_id", "model_name", "model_version"]:
            return
        constraint_name = str(pk_info["constraint_name"])
        self.conn.execute(
            f"ALTER TABLE {table_name} DROP CONSTRAINT IF EXISTS {constraint_name}"
        )
        self.conn.execute(
            f"""
            ALTER TABLE {table_name}
            ADD CONSTRAINT {table_name}_pkey
            PRIMARY KEY (capture_id, model_name, model_version)
            """
        )

    def _ensure_model_specific_vector_indexes(self):
        configured_models = [
            (
                os.getenv("GEOSPY_CLIP_MODEL", "ViT-B-32"),
                os.getenv("GEOSPY_CLIP_VERSION", "open_clip"),
            )
        ]

        place_enabled = (
            os.getenv("GEOSPY_PLACE_MODEL_ENABLED", "0").strip().lower()
            in {"1", "true", "yes", "on"}
        )
        if place_enabled:
            place_runtime = os.getenv("GEOSPY_PLACE_RUNTIME", "open_clip").strip().lower()
            if place_runtime == "open_clip":
                place_model_name = os.getenv("GEOSPY_PLACE_CLIP_MODEL", "ViT-B-16")
                place_model_version = os.getenv(
                    "GEOSPY_PLACE_MODEL_VERSION",
                    "open_clip_place",
                )
            else:
                place_model_name = os.getenv("GEOSPY_PLACE_MODEL_NAME", "")
                place_model_version = os.getenv(
                    "GEOSPY_PLACE_MODEL_VERSION",
                    "place_hf",
                )
            configured_models.append(
                (
                    place_model_name,
                    place_model_version,
                )
            )

        def ensure_indexes_for_table(table_name: str, index_prefix: str, models: Sequence[Tuple[str, str]]) -> None:
            seen = set()
            for model_name, model_version in models:
                key = (str(model_name), str(model_version))
                if key in seen:
                    continue
                seen.add(key)
                slug = (
                    f"{model_name}_{model_version}"
                    .replace("-", "_")
                    .replace(".", "_")
                    .replace("/", "_")
                    .lower()
                )
                slug = "".join(ch for ch in slug if ch.isalnum() or ch == "_")
                if not slug:
                    continue
                index_name = f"{index_prefix}_{slug}"[:60]
                model_name_sql = str(model_name).replace("'", "''")
                model_version_sql = str(model_version).replace("'", "''")
                self.conn.execute(
                    f"""
                    CREATE INDEX IF NOT EXISTS {index_name}
                    ON {table_name}
                    USING ivfflat (embedding vector_cosine_ops)
                    WITH (lists = 100)
                    WHERE model_name = '{model_name_sql}' AND model_version = '{model_version_sql}'
                    """
                )

        ensure_indexes_for_table(
            CAPTURE_EMBEDDINGS_TABLE,
            "idx_cap_embed_ivf",
            configured_models,
        )

    def is_duplicate(self, lat: float, lon: float, radius_meters: float = 25.0) -> bool:
        return self.get_nearby_panorama_id(lat, lon, radius_meters) is not None

    def get_nearby_panorama_id(
        self, lat: float, lon: float, radius_meters: float = 25.0
    ) -> Optional[int]:
        lat_delta = radius_meters / 111320.0
        cos_lat = math.cos(math.radians(lat))
        lon_scale = max(0.01, abs(cos_lat))
        lon_delta = radius_meters / (111320.0 * lon_scale)

        rows = self.conn.execute(
            """
            SELECT id, lat, lon FROM panoramas
            WHERE lat BETWEEN %s AND %s
              AND lon BETWEEN %s AND %s
            """,
            (lat - lat_delta, lat + lat_delta, lon - lon_delta, lon + lon_delta),
        ).fetchall()

        for row in rows:
            dist = self._haversine(lat, lon, row["lat"], row["lon"])
            if dist <= radius_meters:
                return int(row["id"])
        return None

    def get_panorama_id_by_pano_id(self, pano_id: Optional[str]) -> Optional[int]:
        if not pano_id:
            return None
        row = self.conn.execute(
            "SELECT id FROM panoramas WHERE pano_id = %s LIMIT 1", (pano_id,)
        ).fetchone()
        return int(row["id"]) if row else None

    def add_panorama(self, pano: Panorama) -> int:
        try:
            row = self.conn.execute(
                """
                INSERT INTO panoramas (lat, lon, pano_id, heading, pitch, timestamp, source_url, city, notes)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id
                """,
                (
                    pano.lat,
                    pano.lon,
                    pano.pano_id,
                    pano.heading,
                    pano.pitch,
                    pano.timestamp,
                    pano.source_url,
                    pano.city,
                    pano.notes,
                ),
            ).fetchone()
            self.conn.commit()
            return int(row["id"])
        except UniqueViolation:
            self.conn.rollback()
            existing_id = self.get_panorama_id_by_pano_id(pano.pano_id)
            if existing_id is not None:
                return existing_id
            raise
        except Exception:
            self.conn.rollback()
            raise

    def add_panorama_if_new(
        self, pano: Panorama, dedup_radius_meters: float = 25.0
    ) -> Tuple[int, bool]:
        pano_match = self.get_panorama_id_by_pano_id(pano.pano_id)
        if pano_match is not None:
            return pano_match, False

        nearby_match = self.get_nearby_panorama_id(
            pano.lat, pano.lon, dedup_radius_meters
        )
        if nearby_match is not None:
            return nearby_match, False

        return self.add_panorama(pano), True

    def add_capture(self, capture: Capture) -> int:
        row = self.conn.execute(
            """
            INSERT INTO captures (
                panorama_id,
                heading,
                pitch,
                filepath,
                width,
                height,
                capture_profile,
                capture_kind,
                is_black_frame,
                quality_reason,
                brightness_mean,
                flagged_at
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id
            """,
            (
                capture.panorama_id,
                capture.heading,
                capture.pitch,
                capture.filepath,
                capture.width,
                capture.height,
                capture.capture_profile,
                capture.capture_kind,
                int(capture.is_black_frame),
                (capture.quality_reason or "")[:200],
                capture.brightness_mean,
                capture.flagged_at,
            ),
        ).fetchone()
        self.conn.commit()
        return int(row["id"])

    def add_capture_if_missing(
        self,
        capture: Capture,
        heading_tolerance: float = 0.1,
        pitch_tolerance: float = 0.1,
    ) -> Tuple[int, bool]:
        row = self.conn.execute(
            """
            SELECT id
            FROM captures
            WHERE panorama_id = %s
              AND capture_profile = %s
              AND ABS(heading - %s) <= %s
              AND ABS(pitch - %s) <= %s
            ORDER BY id ASC
            LIMIT 1
            """,
            (
                capture.panorama_id,
                capture.capture_profile,
                float(capture.heading),
                float(heading_tolerance),
                float(capture.pitch),
                float(pitch_tolerance),
            ),
        ).fetchone()
        if row:
            return int(row["id"]), False
        return self.add_capture(capture), True

    def mark_capture_quality(
        self,
        capture_id: int,
        is_black_frame: bool,
        quality_reason: str,
        brightness_mean: Optional[float] = None,
    ):
        self.conn.execute(
            """
            UPDATE captures
            SET is_black_frame = %s,
                quality_reason = %s,
                brightness_mean = %s,
                flagged_at = CASE WHEN %s THEN NOW() ELSE flagged_at END
            WHERE id = %s
            """,
            (
                1 if is_black_frame else 0,
                (quality_reason or "")[:200],
                brightness_mean,
                bool(is_black_frame),
                capture_id,
            ),
        )
        self.conn.commit()

    def iter_capture_rows(self) -> Iterable[dict]:
        rows = self.conn.execute(
            "SELECT id, panorama_id, heading, pitch, capture_profile, capture_kind, filepath FROM captures ORDER BY id"
        ).fetchall()
        for row in rows:
            yield self._normalize_row(dict(row))

    def get_all_panoramas(self) -> List[dict]:
        rows = self.conn.execute(
            "SELECT * FROM panoramas ORDER BY created_at DESC"
        ).fetchall()
        return [self._normalize_row(dict(row)) for row in rows]

    def get_captures_for_panorama(self, panorama_id: int) -> List[dict]:
        rows = self.conn.execute(
            "SELECT * FROM captures WHERE panorama_id = %s ORDER BY heading, pitch",
            (panorama_id,),
        ).fetchall()
        return [self._normalize_row(dict(row)) for row in rows]

    def get_total_capture_count(self) -> int:
        row = self.conn.execute("SELECT COUNT(*) AS c FROM captures").fetchone()
        return int(row["c"] if row else 0)

    def get_capture_metadata_for_capture_ids(
        self, capture_ids: Sequence[int]
    ) -> Dict[int, dict]:
        normalized_ids = [int(capture_id) for capture_id in capture_ids if capture_id]
        if not normalized_ids:
            return {}
        rows = self.conn.execute(
            """
            SELECT
                c.id AS capture_id,
                c.panorama_id,
                c.heading,
                c.pitch,
                c.capture_profile,
                c.capture_kind,
                c.filepath,
                p.pano_id,
                p.lat,
                p.lon
            FROM captures c
            JOIN panoramas p ON p.id = c.panorama_id
            WHERE c.id = ANY(%s)
            """,
            (normalized_ids,),
        ).fetchall()
        by_capture_id: Dict[int, dict] = {}
        for row in rows:
            normalized = self._normalize_row(dict(row))
            by_capture_id[int(normalized["capture_id"])] = normalized
        return by_capture_id

    def get_existing_capture_views(
        self, panorama_id: int, capture_profile: Optional[str] = None
    ) -> set[Tuple[float, float]]:
        if capture_profile:
            rows = self.conn.execute(
                """
                SELECT heading, pitch
                FROM captures
                WHERE panorama_id = %s
                  AND capture_profile = %s
                """,
                (panorama_id, capture_profile),
            ).fetchall()
        else:
            rows = self.conn.execute(
                """
                SELECT heading, pitch
                FROM captures
                WHERE panorama_id = %s
                """,
                (panorama_id,),
            ).fetchall()
        return {
            (round(float(r["heading"]), 3), round(float(r.get("pitch", 75.0)), 3))
            for r in rows
        }

    def get_panoramas_in_bbox(
        self, min_lat: float, min_lon: float, max_lat: float, max_lon: float
    ) -> List[dict]:
        rows = self.conn.execute(
            """
            SELECT id, lat, lon, pano_id, heading, pitch, timestamp
            FROM panoramas
            WHERE lat BETWEEN %s AND %s
              AND lon BETWEEN %s AND %s
            ORDER BY id ASC
            """,
            (min_lat, max_lat, min_lon, max_lon),
        ).fetchall()
        return [self._normalize_row(dict(row)) for row in rows]

    def get_missing_views_for_panoramas(
        self,
        panorama_ids: List[int],
        required_views: Sequence[Tuple[float, float]],
        capture_profile: str,
    ) -> Dict[int, List[Tuple[float, float]]]:
        if not panorama_ids:
            return {}
        rounded_required = {
            (round(float(h), 3), round(float(p), 3)) for h, p in required_views
        }
        rows = self.conn.execute(
            """
            SELECT panorama_id, heading, pitch
            FROM captures
            WHERE panorama_id = ANY(%s)
              AND capture_profile = %s
            """,
            (panorama_ids, capture_profile),
        ).fetchall()
        existing_by_panorama: Dict[int, set[Tuple[float, float]]] = {
            int(pid): set() for pid in panorama_ids
        }
        for row in rows:
            pid = int(row["panorama_id"])
            existing_by_panorama.setdefault(pid, set()).add(
                (round(float(row["heading"]), 3), round(float(row.get("pitch", 75.0)), 3))
            )

        result: Dict[int, List[Tuple[float, float]]] = {}
        for panorama_id in panorama_ids:
            existing = existing_by_panorama.get(int(panorama_id), set())
            missing = sorted(list(rounded_required - existing), key=lambda x: (x[1], x[0]))
            result[int(panorama_id)] = missing
        return result

    def is_vector_ready(self) -> bool:
        return bool(self.vector_enabled)

    def get_capture_embedding_stats(
        self, model_name: str, model_version: str, embedding_base: str = EMBEDDING_BASE_CLIP
    ) -> dict:
        total_captures = int(
            self.conn.execute("SELECT COUNT(*) AS c FROM captures").fetchone()["c"]
        )
        if not self.vector_enabled:
            return {
                "vector_enabled": False,
                "model_name": model_name,
                "model_version": model_version,
                "total_captures": total_captures,
                "embedded_captures": 0,
                "pending_captures": total_captures,
            }

        table_name = self._embedding_table_for_base(embedding_base)
        embedded_captures = int(
            self.conn.execute(
                f"""
                SELECT COUNT(*) AS c
                FROM {table_name}
                WHERE model_name = %s AND model_version = %s
                """,
                (model_name, model_version),
            ).fetchone()["c"]
        )
        return {
            "vector_enabled": True,
            "embedding_base": self._normalize_embedding_base(embedding_base),
            "model_name": model_name,
            "model_version": model_version,
            "total_captures": total_captures,
            "embedded_captures": embedded_captures,
            "pending_captures": max(0, total_captures - embedded_captures),
        }

    def list_unembedded_captures(
        self,
        model_name: str,
        model_version: str,
        limit: int = 64,
        after_capture_id: int = 0,
        embedding_base: str = EMBEDDING_BASE_CLIP,
    ) -> List[dict]:
        if not self.vector_enabled:
            return []
        table_name = self._embedding_table_for_base(embedding_base)
        rows = self.conn.execute(
            f"""
            SELECT c.id AS capture_id, c.panorama_id, c.filepath, c.heading
            FROM captures c
            WHERE c.id > %s
              AND NOT EXISTS (
                    SELECT 1
                    FROM {table_name} ce
                    WHERE ce.capture_id = c.id
                      AND ce.model_name = %s
                      AND ce.model_version = %s
              )
            ORDER BY c.id ASC
            LIMIT %s
            """,
            (after_capture_id, model_name, model_version, max(1, int(limit))),
        ).fetchall()
        return [self._normalize_row(dict(row)) for row in rows]

    def list_captures_missing_any_embeddings(
        self,
        models: Sequence[Tuple[str, str]],
        limit: int = 64,
        after_capture_id: int = 0,
        embedding_base: str = EMBEDDING_BASE_CLIP,
    ) -> List[dict]:
        if not self.vector_enabled:
            return []
        table_name = self._embedding_table_for_base(embedding_base)
        normalized_models = [
            (str(model_name), str(model_version))
            for model_name, model_version in models
            if str(model_name).strip()
        ]
        if not normalized_models:
            return []
        values_sql = ", ".join(["(%s, %s)"] * len(normalized_models))
        model_params: List = []
        for model_name, model_version in normalized_models:
            model_params.extend([model_name, model_version])
        query = f"""
            WITH target_models(model_name, model_version) AS (
                VALUES {values_sql}
            )
            SELECT c.id AS capture_id, c.panorama_id, c.filepath, c.heading
            FROM captures c
            WHERE c.id > %s
              AND EXISTS (
                    SELECT 1
                    FROM target_models tm
                    WHERE NOT EXISTS (
                        SELECT 1
                        FROM {table_name} ce
                        WHERE ce.capture_id = c.id
                          AND ce.model_name = tm.model_name
                          AND ce.model_version = tm.model_version
                    )
              )
            ORDER BY c.id ASC
            LIMIT %s
        """
        params: List = [*model_params, int(after_capture_id), max(1, int(limit))]
        rows = self.conn.execute(query, tuple(params)).fetchall()
        return [self._normalize_row(dict(row)) for row in rows]

    def upsert_capture_embedding(
        self,
        capture_id: int,
        model_name: str,
        model_version: str,
        embedding: Sequence[float],
        embedding_base: str = EMBEDDING_BASE_CLIP,
    ):
        if not self.vector_enabled:
            raise RuntimeError("Vector extension is not enabled on this database")
        table_name = self._embedding_table_for_base(embedding_base)
        vector_literal = self._vector_literal(embedding)
        self.conn.execute(
            f"""
            INSERT INTO {table_name} (
                capture_id,
                model_name,
                model_version,
                embedding,
                created_at,
                updated_at
            )
            VALUES (%s, %s, %s, %s::vector, NOW(), NOW())
            ON CONFLICT (capture_id, model_name, model_version) DO UPDATE
            SET embedding = EXCLUDED.embedding,
                updated_at = NOW()
            """,
            (capture_id, model_name, model_version, vector_literal),
        )
        self.conn.commit()

    def upsert_capture_embeddings_batch(
        self,
        capture_vectors: Sequence[Tuple[int, Sequence[float]]],
        model_name: str,
        model_version: str,
        embedding_base: str = EMBEDDING_BASE_CLIP,
    ) -> int:
        if not self.vector_enabled:
            raise RuntimeError("Vector extension is not enabled on this database")
        table_name = self._embedding_table_for_base(embedding_base)
        rows = [
            (int(capture_id), model_name, model_version, self._vector_literal(embedding))
            for capture_id, embedding in capture_vectors
        ]
        if not rows:
            return 0
        with self.conn.cursor() as cur:
            cur.executemany(
                f"""
                INSERT INTO {table_name} (
                    capture_id,
                    model_name,
                    model_version,
                    embedding,
                    created_at,
                    updated_at
                )
                VALUES (%s, %s, %s, %s::vector, NOW(), NOW())
                ON CONFLICT (capture_id, model_name, model_version) DO UPDATE
                SET embedding = EXCLUDED.embedding,
                    updated_at = NOW()
                """,
                rows,
            )
        self.conn.commit()
        return len(rows)

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
        if not self.vector_enabled:
            raise RuntimeError("Vector extension is not enabled on this database")
        table_name = self._embedding_table_for_base(embedding_base)
        vector_literal = self._vector_literal(embedding)
        limit = max(1, min(int(max_top_k), int(top_k)))
        probes = max(
            1,
            min(
                1000,
                int(
                    DEFAULT_RETRIEVAL_IVFFLAT_PROBES
                    if ivfflat_probes is None
                    else ivfflat_probes
                ),
            ),
        )
        savepoint_name = "retrieval_probes_set"
        try:
            self.conn.execute(f"SAVEPOINT {savepoint_name}")
            self.conn.execute(f"SET LOCAL ivfflat.probes = {probes}")
            self.conn.execute(f"RELEASE SAVEPOINT {savepoint_name}")
        except Exception as exc:
            try:
                self.conn.execute(f"ROLLBACK TO SAVEPOINT {savepoint_name}")
                self.conn.execute(f"RELEASE SAVEPOINT {savepoint_name}")
            except Exception:
                pass
            if trace_id:
                log.warning(
                    "retrieval_db trace_id=%s failed to set ivfflat.probes=%s error=%s",
                    trace_id,
                    probes,
                    exc,
                )
        params: List = [vector_literal, model_name, model_version]
        similarity_clause = ""
        if min_similarity is not None:
            similarity_clause = "AND (1 - (ce.embedding <=> %s::vector)) >= %s"
            params.extend([vector_literal, float(min_similarity)])
        params.extend([vector_literal, limit])
        rows = self.conn.execute(
            f"""
            SELECT
                c.id AS capture_id,
                c.panorama_id,
                c.heading,
                c.pitch,
                c.capture_profile,
                c.capture_kind,
                c.filepath,
                p.pano_id,
                p.lat,
                p.lon,
                (1 - (ce.embedding <=> %s::vector)) AS similarity
            FROM {table_name} ce
            JOIN captures c ON c.id = ce.capture_id
            JOIN panoramas p ON p.id = c.panorama_id
            WHERE ce.model_name = %s
              AND ce.model_version = %s
              {similarity_clause}
            ORDER BY ce.embedding <=> %s::vector
            LIMIT %s
            """,
            tuple(params),
        ).fetchall()
        result = [self._normalize_row(dict(row)) for row in rows]
        if trace_id:
            log.info(
                "retrieval_db trace_id=%s base=%s top_k=%s min_similarity=%s returned=%s model=%s version=%s probes=%s",
                trace_id,
                self._normalize_embedding_base(embedding_base),
                limit,
                min_similarity,
                len(result),
                model_name,
                model_version,
                probes,
            )
        return result

    def normalize_capture_filepaths(self, captures_dir: str) -> Dict[str, int]:
        rows = self.conn.execute("SELECT id, filepath FROM captures").fetchall()
        captures_abs = os.path.abspath(captures_dir)
        updates: List[Tuple[str, int]] = []
        unchanged = 0

        for row in rows:
            raw = (row["filepath"] or "").strip()
            if not raw:
                unchanged += 1
                continue

            unix_raw = raw.replace("\\", "/")
            new_path: Optional[str] = None

            if unix_raw.startswith("captures/"):
                unchanged += 1
                continue

            if unix_raw.startswith("/captures/"):
                new_path = unix_raw.lstrip("/")
            elif os.path.isabs(raw):
                abs_raw = os.path.abspath(raw)
                if abs_raw.startswith(captures_abs + os.sep) or abs_raw == captures_abs:
                    rel_inside = os.path.relpath(abs_raw, captures_abs).replace("\\", "/")
                    if rel_inside != ".":
                        new_path = f"captures/{rel_inside}"
            elif "/captures/" in unix_raw:
                tail = unix_raw.split("/captures/", 1)[1].lstrip("/")
                if tail:
                    new_path = f"captures/{tail}"

            if not new_path:
                unchanged += 1
                continue

            updates.append((new_path, int(row["id"])))

        if updates:
            with self.conn.cursor() as cur:
                cur.executemany(
                    "UPDATE captures SET filepath = %s WHERE id = %s",
                    updates,
                )
            self.conn.commit()

        return {"updated": len(updates), "unchanged": unchanged, "total": len(rows)}

    def get_stats(self) -> dict:
        pano_count = int(
            self.conn.execute("SELECT COUNT(*) AS c FROM panoramas").fetchone()["c"]
        )
        capture_count = int(
            self.conn.execute("SELECT COUNT(*) AS c FROM captures").fetchone()["c"]
        )

        bounds = self.conn.execute(
            """
            SELECT
                MIN(lat) AS min_lat,
                MAX(lat) AS max_lat,
                MIN(lon) AS min_lon,
                MAX(lon) AS max_lon
            FROM panoramas
            """
        ).fetchone()

        return {
            "total_panoramas": pano_count,
            "total_captures": capture_count,
            "bounds": {
                "min_lat": bounds["min_lat"],
                "max_lat": bounds["max_lat"],
                "min_lon": bounds["min_lon"],
                "max_lon": bounds["max_lon"],
            }
            if bounds and bounds["min_lat"] is not None
            else None,
        }

    def get_panoramas_geojson(self) -> dict:
        rows = self.conn.execute(
            """
            SELECT
                p.id,
                p.lat,
                p.lon,
                p.pano_id,
                p.heading,
                p.timestamp,
                p.created_at,
                COUNT(c.id) AS capture_count,
                STRING_AGG(c.filepath, '|||' ORDER BY c.heading) AS capture_paths
            FROM panoramas p
            LEFT JOIN captures c ON c.panorama_id = p.id
            GROUP BY p.id, p.lat, p.lon, p.pano_id, p.heading, p.timestamp, p.created_at
            ORDER BY p.created_at DESC
            """
        ).fetchall()

        features = []
        for p in rows:
            captures = p["capture_paths"].split("|||") if p["capture_paths"] else []
            ts = p["timestamp"]
            if isinstance(ts, (datetime, date)):
                ts = ts.isoformat()
            features.append(
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "Point",
                        "coordinates": [p["lon"], p["lat"]],
                    },
                    "properties": {
                        "id": p["id"],
                        "pano_id": p["pano_id"],
                        "heading": p["heading"],
                        "timestamp": ts,
                        "capture_count": p["capture_count"],
                        "captures": captures,
                    },
                }
            )

        return {"type": "FeatureCollection", "features": features}

    def get_panoramas_bbox_points(
        self,
        min_lat: float,
        min_lon: float,
        max_lat: float,
        max_lon: float,
        limit: int = 6000,
    ) -> List[dict]:
        rows = self.conn.execute(
            """
            SELECT
                p.id,
                p.lat,
                p.lon,
                p.pano_id,
                p.heading,
                p.timestamp
            FROM panoramas p
            WHERE p.lat BETWEEN %s AND %s
              AND p.lon BETWEEN %s AND %s
            LIMIT %s
            """,
            (
                float(min_lat),
                float(max_lat),
                float(min_lon),
                float(max_lon),
                max(1, int(limit)),
            ),
        ).fetchall()
        return [self._normalize_row(dict(row)) for row in rows]

    def get_panoramas_bbox_clusters(
        self,
        min_lat: float,
        min_lon: float,
        max_lat: float,
        max_lon: float,
        zoom: int,
        limit: int = 2000,
    ) -> List[dict]:
        min_lat = float(min_lat)
        min_lon = float(min_lon)
        max_lat = float(max_lat)
        max_lon = float(max_lon)
        zoom = max(1, int(zoom))
        span_lat = max(1e-6, max_lat - min_lat)
        span_lon = max(1e-6, max_lon - min_lon)
        target_cells = max(20, min(900, int((zoom + 2) ** 2)))
        side = math.sqrt(target_cells)
        bucket_lat = max(0.00005, span_lat / side)
        bucket_lon = max(0.00005, span_lon / side)
        rows = self.conn.execute(
            """
            SELECT
                FLOOR((p.lat - %s) / %s) AS gy,
                FLOOR((p.lon - %s) / %s) AS gx,
                COUNT(*) AS point_count,
                AVG(p.lat) AS lat,
                AVG(p.lon) AS lon,
                MAX(p.created_at) AS newest_ts,
                MIN(p.id) AS sample_panorama_id
            FROM panoramas p
            WHERE p.lat BETWEEN %s AND %s
              AND p.lon BETWEEN %s AND %s
            GROUP BY gy, gx
            ORDER BY point_count DESC
            LIMIT %s
            """,
            (
                min_lat,
                bucket_lat,
                min_lon,
                bucket_lon,
                min_lat,
                max_lat,
                min_lon,
                max_lon,
                max(1, int(limit)),
            ),
        ).fetchall()
        clusters: List[dict] = []
        for row in rows:
            item = self._normalize_row(dict(row))
            item["bucket_lat"] = bucket_lat
            item["bucket_lon"] = bucket_lon
            clusters.append(item)
        return clusters

    def queue_seed_points(self, points: Iterable[Tuple[float, float]]) -> int:
        before = int(
            self.conn.execute("SELECT COUNT(*) AS c FROM seed_tasks").fetchone()["c"]
        )
        rows = [(round(lat, 6), round(lon, 6)) for lat, lon in points]
        if rows:
            with self.conn.cursor() as cur:
                cur.executemany(
                    "INSERT INTO seed_tasks (lat, lon) VALUES (%s, %s) ON CONFLICT (lat, lon) DO NOTHING",
                    rows,
                )
            self.conn.commit()
        after = int(
            self.conn.execute("SELECT COUNT(*) AS c FROM seed_tasks").fetchone()["c"]
        )
        return after - before

    def clear_seed_tasks(self):
        self.conn.execute("DELETE FROM seed_tasks")
        self.conn.commit()

    def claim_next_seed(
        self, worker_id: str, lease_seconds: int = 300
    ) -> Optional[Dict[str, float]]:
        lease_seconds = max(1, int(lease_seconds))
        try:
            row = self.conn.execute(
                """
                WITH candidate AS (
                    SELECT id
                    FROM seed_tasks
                    WHERE status = 'pending'
                       OR (
                            status = 'in_progress'
                            AND claimed_at <= NOW() - (%s * INTERVAL '1 second')
                       )
                    ORDER BY
                        CASE WHEN status = 'pending' THEN 0 ELSE 1 END,
                        attempts ASC,
                        id ASC
                    FOR UPDATE SKIP LOCKED
                    LIMIT 1
                )
                UPDATE seed_tasks st
                SET status = 'in_progress',
                    claimed_by = %s,
                    claimed_at = NOW(),
                    attempts = st.attempts + 1,
                    updated_at = NOW()
                FROM candidate c
                WHERE st.id = c.id
                RETURNING st.id, st.lat, st.lon, st.attempts
                """,
                (lease_seconds, worker_id),
            ).fetchone()
            self.conn.commit()

            if row is None:
                return None
            return {
                "id": int(row["id"]),
                "lat": float(row["lat"]),
                "lon": float(row["lon"]),
                "attempts": int(row["attempts"]),
            }
        except Exception:
            self.conn.rollback()
            raise

    def mark_seed_status(self, task_id: int, status: str, last_error: str = ""):
        if status not in {"done", "skipped", "failed"}:
            raise ValueError(f"Invalid task status: {status}")

        self.conn.execute(
            """
            UPDATE seed_tasks
            SET status = %s,
                claimed_by = NULL,
                claimed_at = NULL,
                last_error = %s,
                updated_at = NOW()
            WHERE id = %s
            """,
            (status, last_error[:500], task_id),
        )
        self.conn.commit()

    def get_seed_task_stats(self) -> dict:
        rows = self.conn.execute(
            """
            SELECT status, COUNT(*) AS count
            FROM seed_tasks
            GROUP BY status
            """
        ).fetchall()
        counts = {row["status"]: int(row["count"]) for row in rows}
        total = int(self.conn.execute("SELECT COUNT(*) AS c FROM seed_tasks").fetchone()["c"])
        return {
            "total": total,
            "pending": counts.get("pending", 0),
            "in_progress": counts.get("in_progress", 0),
            "done": counts.get("done", 0),
            "skipped": counts.get("skipped", 0),
            "failed": counts.get("failed", 0),
        }

    @staticmethod
    def _vector_literal(values: Sequence[float]) -> str:
        return "[" + ",".join(f"{float(v):.8f}" for v in values) + "]"

    @staticmethod
    def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        r = 6371000
        phi1 = math.radians(lat1)
        phi2 = math.radians(lat2)
        dphi = math.radians(lat2 - lat1)
        dlam = math.radians(lon2 - lon1)
        a = (
            math.sin(dphi / 2) ** 2
            + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
        )
        return r * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    def close(self):
        self.conn.close()

    @staticmethod
    def _normalize_row(row: Dict) -> Dict:
        normalized = {}
        for key, value in row.items():
            if isinstance(value, (datetime, date)):
                normalized[key] = value.isoformat()
            else:
                normalized[key] = value
        return normalized
