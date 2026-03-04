"""Migrate capture_embeddings to multi-model primary key and refresh ANN indexes."""

import argparse

from config import CrawlerConfig
from db.postgres_database import Database


def main():
    parser = argparse.ArgumentParser(
        description="Migrate capture_embeddings schema for multi-model retrieval."
    )
    parser.add_argument(
        "--reindex",
        action="store_true",
        help="Run REINDEX TABLE and ANALYZE capture_embeddings after migration.",
    )
    args = parser.parse_args()

    cfg = CrawlerConfig()
    db = Database(cfg.DATABASE_URL)
    try:
        if not db.is_vector_ready():
            raise SystemExit("Vector extension is unavailable.")
        # Re-run migration helper to ensure old single-key layouts are upgraded.
        db._migrate_capture_embeddings_primary_key("capture_embeddings")
        db._ensure_model_specific_vector_indexes()
        db.conn.commit()
        if args.reindex:
            db.conn.execute("REINDEX TABLE capture_embeddings")
            db.conn.execute("ANALYZE capture_embeddings")
            db.conn.commit()
        print("capture_embeddings schema migration complete.")
    finally:
        db.close()


if __name__ == "__main__":
    main()
