"""
Create a local backup dump from the Docker Postgres container.

Usage:
    python -m utils.backup_postgres
    python -m utils.backup_postgres --output-dir backups
    python -m utils.backup_postgres --container geospy-postgres --db geospy --user geospy
"""

import argparse
import datetime as dt
import gzip
import os
import subprocess
from pathlib import Path


def _default_name(prefix: str, gzip_enabled: bool) -> str:
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    ext = ".sql.gz" if gzip_enabled else ".sql"
    return f"{prefix}_{ts}{ext}"


def _dump_from_container(
    container: str,
    db_name: str,
    db_user: str,
    output_path: Path,
    gzip_enabled: bool,
):
    cmd = [
        "docker",
        "exec",
        "-i",
        container,
        "pg_dump",
        "-U",
        db_user,
        "-d",
        db_name,
        "--format=plain",
        "--no-owner",
        "--no-privileges",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.stdout is None:
        raise RuntimeError("Failed to open pg_dump output stream")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        if gzip_enabled:
            with gzip.open(output_path, "wb") as f:
                for chunk in iter(lambda: proc.stdout.read(1024 * 1024), b""):
                    if chunk:
                        f.write(chunk)
        else:
            with open(output_path, "wb") as f:
                for chunk in iter(lambda: proc.stdout.read(1024 * 1024), b""):
                    if chunk:
                        f.write(chunk)
    finally:
        proc.stdout.close()

    stderr = proc.stderr.read().decode("utf-8", errors="replace") if proc.stderr else ""
    rc = proc.wait()
    if rc != 0:
        if output_path.exists():
            output_path.unlink(missing_ok=True)
        raise RuntimeError(f"pg_dump failed (exit {rc}): {stderr.strip()}")


def main():
    parser = argparse.ArgumentParser(description="Backup Docker Postgres DB to local file.")
    parser.add_argument("--container", type=str, default="geospy-postgres", help="Docker container name")
    parser.add_argument("--db", type=str, default=os.getenv("POSTGRES_DB", "geospy"), help="Database name")
    parser.add_argument("--user", type=str, default=os.getenv("POSTGRES_USER", "geospy"), help="Database user")
    parser.add_argument("--output-dir", type=str, default="backups", help="Directory to write backup files")
    parser.add_argument("--prefix", type=str, default="geospy_postgres", help="Backup filename prefix")
    parser.add_argument("--no-gzip", action="store_true", help="Write plain .sql instead of .sql.gz")
    args = parser.parse_args()

    gzip_enabled = not args.no_gzip
    out_name = _default_name(args.prefix, gzip_enabled)
    out_path = Path(args.output_dir) / out_name

    _dump_from_container(
        container=args.container,
        db_name=args.db,
        db_user=args.user,
        output_path=out_path,
        gzip_enabled=gzip_enabled,
    )

    print(f"Backup written: {out_path}")


if __name__ == "__main__":
    main()
