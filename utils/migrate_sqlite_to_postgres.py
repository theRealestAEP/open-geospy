"""
One-time migration from legacy SQLite DB to Postgres.

Usage:
    python -m utils.migrate_sqlite_to_postgres --dry-run
    python -m utils.migrate_sqlite_to_postgres --apply
    python -m utils.migrate_sqlite_to_postgres --apply --truncate-target
"""

import argparse
import os
import sqlite3
from typing import Dict, List

from db.postgres_database import Database


DEFAULT_SQLITE_PATH = "db/locations.db"
DEFAULT_DATABASE_URL = os.getenv(
    "DATABASE_URL", "postgresql://geospy:geospy@127.0.0.1:5432/geospy"
)


def _fetch_sqlite_rows(conn: sqlite3.Connection, table: str) -> List[Dict]:
    rows = conn.execute(f"SELECT * FROM {table} ORDER BY id").fetchall()
    return [dict(row) for row in rows]


def _source_counts(conn: sqlite3.Connection) -> Dict[str, int]:
    return {
        "panoramas": int(conn.execute("SELECT COUNT(*) FROM panoramas").fetchone()[0]),
        "captures": int(conn.execute("SELECT COUNT(*) FROM captures").fetchone()[0]),
        "seed_tasks": int(conn.execute("SELECT COUNT(*) FROM seed_tasks").fetchone()[0]),
    }


def _target_counts(db: Database) -> Dict[str, int]:
    return {
        "panoramas": int(db.conn.execute("SELECT COUNT(*) AS c FROM panoramas").fetchone()["c"]),
        "captures": int(db.conn.execute("SELECT COUNT(*) AS c FROM captures").fetchone()["c"]),
        "seed_tasks": int(db.conn.execute("SELECT COUNT(*) AS c FROM seed_tasks").fetchone()["c"]),
    }


def _truncate_target(db: Database):
    db.conn.execute("TRUNCATE TABLE seed_tasks RESTART IDENTITY CASCADE")
    db.conn.execute("TRUNCATE TABLE captures RESTART IDENTITY CASCADE")
    db.conn.execute("TRUNCATE TABLE panoramas RESTART IDENTITY CASCADE")
    db.conn.commit()


def _set_sequences(db: Database):
    for table in ("panoramas", "captures", "seed_tasks"):
        db.conn.execute(
            f"""
            SELECT setval(
                pg_get_serial_sequence('{table}', 'id'),
                COALESCE((SELECT MAX(id) FROM {table}), 1),
                true
            )
            """
        )
    db.conn.commit()


def _migrate_panoramas(db: Database, rows: List[Dict]) -> int:
    migrated = 0
    seen_pano_ids = set()
    duplicate_pano_ids = 0
    for row in rows:
        pano_id = row.get("pano_id")
        if pano_id:
            if pano_id in seen_pano_ids:
                pano_id = None
                duplicate_pano_ids += 1
            else:
                seen_pano_ids.add(pano_id)
        db.conn.execute(
            """
            INSERT INTO panoramas (id, lat, lon, pano_id, heading, pitch, timestamp, source_url, city, notes, created_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (id) DO UPDATE SET
                lat = EXCLUDED.lat,
                lon = EXCLUDED.lon,
                pano_id = EXCLUDED.pano_id,
                heading = EXCLUDED.heading,
                pitch = EXCLUDED.pitch,
                timestamp = EXCLUDED.timestamp,
                source_url = EXCLUDED.source_url,
                city = EXCLUDED.city,
                notes = EXCLUDED.notes,
                created_at = EXCLUDED.created_at
            """,
            (
                int(row["id"]),
                float(row["lat"]),
                float(row["lon"]),
                pano_id,
                float(row.get("heading") or 0.0),
                float(row.get("pitch") or 0.0),
                row["timestamp"],
                row.get("source_url"),
                row.get("city") or "",
                row.get("notes") or "",
                row.get("created_at"),
            ),
        )
        migrated += 1
    db.conn.commit()
    if duplicate_pano_ids:
        print(f"Note: cleared duplicate pano_id on {duplicate_pano_ids} panorama rows during migration.")
    return migrated


def _migrate_captures(db: Database, rows: List[Dict]) -> int:
    migrated = 0
    for row in rows:
        db.conn.execute(
            """
            INSERT INTO captures (
                id, panorama_id, heading, filepath, width, height, created_at,
                is_black_frame, quality_reason, brightness_mean, flagged_at
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (id) DO UPDATE SET
                panorama_id = EXCLUDED.panorama_id,
                heading = EXCLUDED.heading,
                filepath = EXCLUDED.filepath,
                width = EXCLUDED.width,
                height = EXCLUDED.height,
                created_at = EXCLUDED.created_at,
                is_black_frame = EXCLUDED.is_black_frame,
                quality_reason = EXCLUDED.quality_reason,
                brightness_mean = EXCLUDED.brightness_mean,
                flagged_at = EXCLUDED.flagged_at
            """,
            (
                int(row["id"]),
                int(row["panorama_id"]),
                float(row.get("heading") or 0.0),
                row["filepath"],
                row.get("width"),
                row.get("height"),
                row.get("created_at"),
                int(row.get("is_black_frame") or 0),
                row.get("quality_reason") or "",
                row.get("brightness_mean"),
                row.get("flagged_at"),
            ),
        )
        migrated += 1
    db.conn.commit()
    return migrated


def _migrate_seed_tasks(db: Database, rows: List[Dict]) -> int:
    migrated = 0
    for row in rows:
        db.conn.execute(
            """
            INSERT INTO seed_tasks (
                id, lat, lon, status, claimed_by, claimed_at, attempts,
                last_error, created_at, updated_at
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (id) DO UPDATE SET
                lat = EXCLUDED.lat,
                lon = EXCLUDED.lon,
                status = EXCLUDED.status,
                claimed_by = EXCLUDED.claimed_by,
                claimed_at = EXCLUDED.claimed_at,
                attempts = EXCLUDED.attempts,
                last_error = EXCLUDED.last_error,
                created_at = EXCLUDED.created_at,
                updated_at = EXCLUDED.updated_at
            """,
            (
                int(row["id"]),
                float(row["lat"]),
                float(row["lon"]),
                row.get("status") or "pending",
                row.get("claimed_by"),
                row.get("claimed_at"),
                int(row.get("attempts") or 0),
                (row.get("last_error") or "")[:500],
                row.get("created_at"),
                row.get("updated_at"),
            ),
        )
        migrated += 1
    db.conn.commit()
    return migrated


def main():
    parser = argparse.ArgumentParser(description="Migrate legacy SQLite DB to Postgres.")
    parser.add_argument("--sqlite", type=str, default=DEFAULT_SQLITE_PATH, help="Source SQLite path")
    parser.add_argument(
        "--postgres",
        type=str,
        default=DEFAULT_DATABASE_URL,
        help="Target Postgres DATABASE_URL",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print summary only, no writes")
    parser.add_argument("--apply", action="store_true", help="Apply migration writes")
    parser.add_argument(
        "--truncate-target",
        action="store_true",
        help="Truncate target tables before migrating",
    )
    args = parser.parse_args()

    if not os.path.exists(args.sqlite):
        raise SystemExit(f"SQLite DB not found: {args.sqlite}")

    sqlite_conn = sqlite3.connect(args.sqlite)
    sqlite_conn.row_factory = sqlite3.Row
    source = _source_counts(sqlite_conn)

    target_db = Database(args.postgres)
    target = _target_counts(target_db)

    print(f"SQLite source:   {args.sqlite}")
    print(f"Postgres target: {args.postgres}")
    print("Source counts:", source)
    print("Target counts:", target)

    if not args.apply:
        print("Dry run only. Re-run with --apply to migrate.")
        sqlite_conn.close()
        target_db.close()
        return

    if args.truncate_target:
        _truncate_target(target_db)
        print("Target tables truncated.")
    else:
        target_now = _target_counts(target_db)
        if any(target_now[k] > 0 for k in ("panoramas", "captures", "seed_tasks")):
            raise SystemExit(
                "Target Postgres has existing data. Re-run with --truncate-target or empty target DB first."
            )

    panoramas = _fetch_sqlite_rows(sqlite_conn, "panoramas")
    captures = _fetch_sqlite_rows(sqlite_conn, "captures")
    seed_tasks = _fetch_sqlite_rows(sqlite_conn, "seed_tasks")

    pano_migrated = _migrate_panoramas(target_db, panoramas)
    captures_migrated = _migrate_captures(target_db, captures)
    seeds_migrated = _migrate_seed_tasks(target_db, seed_tasks)
    _set_sequences(target_db)

    final_counts = _target_counts(target_db)
    print("Migrated rows:")
    print(f"  panoramas: {pano_migrated}")
    print(f"  captures:  {captures_migrated}")
    print(f"  seed_tasks:{seeds_migrated}")
    print("Final target counts:", final_counts)
    print("Migration complete.")

    sqlite_conn.close()
    target_db.close()


if __name__ == "__main__":
    main()
