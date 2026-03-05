#!/usr/bin/env python3
"""Sync capture_embeddings rows from Postgres (pgvector) into a LanceDB table."""

import argparse
import logging
import os
import sys
import time
from typing import Dict, List, Optional

import psycopg
import pyarrow as pa
from psycopg.rows import dict_row


log = logging.getLogger("sync_pgvector_to_lancedb")


def _parse_embedding_vector(raw: str) -> List[float]:
    payload = (raw or "").strip()
    if payload.startswith("[") and payload.endswith("]"):
        payload = payload[1:-1]
    if not payload:
        return []
    return [float(item) for item in payload.split(",")]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Copy capture_embeddings from Postgres into LanceDB."
    )
    parser.add_argument(
        "--db-url",
        default=os.getenv("DATABASE_URL", ""),
        help="Postgres connection string (default: DATABASE_URL env)",
    )
    parser.add_argument(
        "--lance-uri",
        default=os.getenv("GEOSPY_LANCEDB_URI", ".lancedb"),
        help="LanceDB URI/path (default: GEOSPY_LANCEDB_URI or .lancedb)",
    )
    parser.add_argument(
        "--table",
        default=os.getenv("GEOSPY_LANCEDB_TABLE", "capture_embeddings"),
        help="Lance table name",
    )
    parser.add_argument(
        "--vector-column",
        default=os.getenv("GEOSPY_LANCEDB_VECTOR_COLUMN", "embedding"),
        help="Vector column name in Lance table",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2000,
        help="Rows per batch for export/import",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional max rows to copy (0 = all rows)",
    )
    parser.add_argument(
        "--mode",
        choices=["overwrite", "append"],
        default="overwrite",
        help="overwrite: recreate table, append: add rows",
    )
    parser.add_argument(
        "--create-index",
        action="store_true",
        help="Build/rebuild Lance vector index after copy",
    )
    parser.add_argument(
        "--num-sub-vectors",
        type=int,
        default=0,
        help=(
            "Optional IVF_PQ num_sub_vectors. Must divide embedding dimension. "
            "If 0, auto-select a valid divisor."
        ),
    )
    return parser


def _connect_lance(uri: str):
    try:
        import lancedb
    except ImportError as exc:
        raise RuntimeError(
            "Missing lancedb dependency. Install with: pip install lancedb"
        ) from exc
    kwargs: Dict[str, str] = {}
    if uri.startswith("db://"):
        api_key = os.getenv("LANCEDB_API_KEY", "").strip()
        if api_key:
            kwargs["api_key"] = api_key
        region = os.getenv("LANCEDB_REGION", "").strip()
        if region:
            kwargs["region"] = region
        host_override = os.getenv("LANCEDB_HOST_OVERRIDE", "").strip()
        if host_override:
            kwargs["host_override"] = host_override
    return lancedb.connect(uri, **kwargs)


def _choose_num_sub_vectors(embedding_dim: int, requested: int = 0) -> int:
    dim = max(1, int(embedding_dim))
    req = int(requested or 0)
    if req > 0:
        if dim % req != 0:
            raise ValueError(
                f"Requested num_sub_vectors={req} does not divide embedding_dim={dim}"
            )
        return req
    # Prefer Lance default if valid; otherwise pick the largest common divisor <= 96.
    if dim % 96 == 0:
        return 96
    for candidate in range(min(95, dim), 1, -1):
        if dim % candidate == 0:
            return candidate
    return 1


def _schema(vector_column: str, dim: int) -> pa.Schema:
    return pa.schema(
        [
            pa.field("capture_id", pa.int64(), nullable=False),
            pa.field("model_name", pa.string(), nullable=False),
            pa.field("model_version", pa.string(), nullable=False),
            pa.field(vector_column, pa.list_(pa.float32(), dim), nullable=False),
        ]
    )


def _open_or_create_lance_table(
    lance_conn,
    *,
    table_name: str,
    mode: str,
    batch: List[dict],
    vector_column: str,
    embedding_dim: int,
):
    names = set(lance_conn.table_names())
    if mode == "overwrite":
        return lance_conn.create_table(
            table_name,
            data=batch,
            schema=_schema(vector_column, embedding_dim),
            mode="overwrite",
        )
    if table_name in names:
        table = lance_conn.open_table(table_name)
        table.add(batch)
        return table
    return lance_conn.create_table(
        table_name,
        data=batch,
        schema=_schema(vector_column, embedding_dim),
    )


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if not args.db_url:
        parser.error("--db-url is required (or set DATABASE_URL)")

    started = time.time()
    copied = 0
    skipped = 0
    embedding_dim: Optional[int] = None
    lance_table = None

    lance_conn = _connect_lance(args.lance_uri)

    log.info(
        "Starting copy postgres->lance db_url=%s lance_uri=%s table=%s mode=%s batch_size=%s",
        args.db_url,
        args.lance_uri,
        args.table,
        args.mode,
        args.batch_size,
    )

    query = """
        SELECT capture_id, model_name, model_version, embedding::text AS embedding_text
        FROM capture_embeddings
        ORDER BY capture_id ASC, model_name ASC, model_version ASC
    """
    if int(args.limit) > 0:
        query += " LIMIT %s"
        query_params = (int(args.limit),)
    else:
        query_params = ()

    with psycopg.connect(args.db_url, row_factory=dict_row) as pg_conn:
        with pg_conn.cursor(name="pg_to_lance_embeddings") as cursor:
            cursor.itersize = max(100, int(args.batch_size))
            cursor.execute(query, query_params)
            while True:
                rows = cursor.fetchmany(max(100, int(args.batch_size)))
                if not rows:
                    break

                payload: List[dict] = []
                for row in rows:
                    vector = _parse_embedding_vector(str(row.get("embedding_text") or ""))
                    if not vector:
                        skipped += 1
                        continue
                    if embedding_dim is None:
                        embedding_dim = len(vector)
                        log.info("Detected embedding dimension=%s", embedding_dim)
                    elif len(vector) != embedding_dim:
                        skipped += 1
                        continue
                    payload.append(
                        {
                            "capture_id": int(row["capture_id"]),
                            "model_name": str(row["model_name"]),
                            "model_version": str(row["model_version"]),
                            args.vector_column: [float(v) for v in vector],
                        }
                    )

                if not payload:
                    continue

                if lance_table is None:
                    lance_table = _open_or_create_lance_table(
                        lance_conn,
                        table_name=args.table,
                        mode=args.mode,
                        batch=payload,
                        vector_column=args.vector_column,
                        embedding_dim=int(embedding_dim or 0),
                    )
                else:
                    lance_table.add(payload)

                copied += len(payload)
                if copied % max(10000, int(args.batch_size) * 5) == 0:
                    elapsed = max(1e-6, time.time() - started)
                    log.info(
                        "Progress copied=%s skipped=%s rate_rows_per_s=%.1f",
                        copied,
                        skipped,
                        copied / elapsed,
                    )

    if lance_table is None:
        log.warning("No rows copied; Lance table was not created.")
        return 0

    if args.create_index:
        selected_num_sub_vectors = _choose_num_sub_vectors(
            int(embedding_dim or 0), int(args.num_sub_vectors)
        )
        log.info(
            (
                "Building Lance vector index table=%s vector_column=%s metric=cosine "
                "embedding_dim=%s num_sub_vectors=%s"
            ),
            args.table,
            args.vector_column,
            int(embedding_dim or 0),
            selected_num_sub_vectors,
        )
        lance_table.create_index(
            metric="cosine",
            vector_column_name=args.vector_column,
            num_sub_vectors=selected_num_sub_vectors,
            replace=True,
        )

    elapsed = time.time() - started
    log.info(
        "Completed copy copied=%s skipped=%s elapsed_s=%.1f rows_per_s=%.1f",
        copied,
        skipped,
        elapsed,
        copied / max(1e-6, elapsed),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
