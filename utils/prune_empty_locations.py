"""
Prune panorama rows that have no capture rows.

Usage:
    python -m utils.prune_empty_locations --dry-run
    python -m utils.prune_empty_locations --apply
    python -m utils.prune_empty_locations --apply --db postgresql://geospy:geospy@127.0.0.1:5432/geospy
"""

import argparse
import os

from db.postgres_database import Database


def find_empty_panorama_rows(db: Database):
    rows = db.conn.execute(
        """
        SELECT
            p.id,
            p.pano_id,
            p.lat,
            p.lon
        FROM panoramas p
        LEFT JOIN captures c ON c.panorama_id = p.id
        WHERE c.id IS NULL
        ORDER BY p.id
        """
    ).fetchall()
    return [dict(row) for row in rows]


def delete_panoramas_by_ids(db: Database, panorama_ids):
    if not panorama_ids:
        return 0
    placeholders = ",".join("?" for _ in panorama_ids)
    cursor = db.conn.execute(
        f"DELETE FROM panoramas WHERE id IN ({placeholders})",
        tuple(int(pid) for pid in panorama_ids),
    )
    db.conn.commit()
    return int(cursor.rowcount or 0)


def main():
    parser = argparse.ArgumentParser(
        description="Delete panorama rows that have no captures."
    )
    parser.add_argument(
        "--db",
        type=str,
        default=os.getenv("DATABASE_URL", "postgresql://geospy:geospy@127.0.0.1:5432/geospy"),
        help="Postgres DATABASE_URL",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply deletions. Without this flag the script is dry-run.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview only (same as default behavior).",
    )
    parser.add_argument(
        "--print-limit",
        type=int,
        default=20,
        help="How many candidate rows to print in preview output.",
    )
    args = parser.parse_args()

    apply_changes = bool(args.apply) and not bool(args.dry_run)
    db = Database(args.db)

    before_stats = db.get_stats()
    empty_rows = find_empty_panorama_rows(db)
    empty_ids = [row["id"] for row in empty_rows]

    print(f"DB: {args.db}")
    print(f"Total panoramas before: {before_stats['total_panoramas']}")
    print(f"Total captures before:  {before_stats['total_captures']}")
    print(f"Panoramas with no captures: {len(empty_rows)}")

    for row in empty_rows[: max(0, int(args.print_limit))]:
        print(
            f"  id={row['id']} pano_id={row.get('pano_id') or 'N/A'} "
            f"lat={row['lat']:.6f} lon={row['lon']:.6f}"
        )
    if len(empty_rows) > max(0, int(args.print_limit)):
        hidden = len(empty_rows) - max(0, int(args.print_limit))
        print(f"  ... and {hidden} more")

    if not apply_changes:
        print("Dry run only. Re-run with --apply to delete these rows.")
        db.close()
        return

    deleted = delete_panoramas_by_ids(db, empty_ids)
    after_stats = db.get_stats()
    db.close()

    print(f"Deleted panoramas:      {deleted}")
    print(f"Total panoramas after:  {after_stats['total_panoramas']}")
    print(f"Total captures after:   {after_stats['total_captures']}")
    print("Done.")


if __name__ == "__main__":
    main()

