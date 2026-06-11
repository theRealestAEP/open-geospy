"""Prep and run an upload of the local captures dataset to a Modal Volume.

Millions of small files upload terribly one-at-a-time, so this packs panorama
directories into uncompressed tar shards, uploads each shard to a staging path
on the volume, then untars it server-side via a small Modal function.

Workflow:
    1. plan    -- local only, no network. Scans the captures dir, groups pano
                  dirs into ~N GiB shards, writes a resumable manifest.
    2. upload  -- tars + uploads + extracts pending shards, updating the
                  manifest after each (safe to interrupt and re-run).
    3. status  -- prints manifest progress.

Examples:
    python -m utils.modal_volume_upload plan
    python -m utils.modal_volume_upload upload --limit 1   # smoke-test 1 shard
    python -m utils.modal_volume_upload upload             # everything pending
    python -m utils.modal_volume_upload status
"""

import argparse
import json
import os
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional

try:
    from env_bootstrap import load_project_env
except ModuleNotFoundError:
    load_project_env = None

if load_project_env is not None:
    load_project_env()

import modal

VOLUME_NAME = os.getenv("GEOSPY_MODAL_CAPTURES_VOLUME", "geospy-captures")
VOLUME_MOUNT = "/vol"
STAGING_DIR = "_staging"
EXTRACT_ROOT = "captures"
DEFAULT_ENVIRONMENT = (
    os.getenv("GEOSPY_MODAL_EMBED_ENVIRONMENT")
    or os.getenv("MODAL_ENVIRONMENT")
    or "google-map-walkers"
)
DEFAULT_MANIFEST = os.path.join("tmp", "modal_upload", "manifest.json")

app = modal.App("geospy-captures-upload")
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
extract_image = modal.Image.debian_slim(python_version="3.12")


@app.function(image=extract_image, volumes={VOLUME_MOUNT: volume}, timeout=3600, cpu=2.0)
def extract_shard(tar_name: str) -> dict:
    import tarfile

    tar_path = os.path.join(VOLUME_MOUNT, STAGING_DIR, tar_name)
    dest = os.path.join(VOLUME_MOUNT, EXTRACT_ROOT)
    os.makedirs(dest, exist_ok=True)
    extracted = 0
    with tarfile.open(tar_path) as tf:
        for member in tf:
            tf.extract(member, dest, filter="data")
            extracted += 1
    os.remove(tar_path)
    volume.commit()
    return {"tar_name": tar_name, "extracted": extracted}


def _load_manifest(path: str) -> dict:
    with open(path) as fh:
        return json.load(fh)


_manifest_lock = threading.Lock()


def _save_manifest(path: str, manifest: dict) -> None:
    with _manifest_lock:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        tmp_path = path + ".tmp"
        with open(tmp_path, "w") as fh:
            json.dump(manifest, fh)
        os.replace(tmp_path, path)


def _sample_mean_file_bytes(captures_dir: str, dirs: List[str], samples: int = 2000) -> float:
    sizes: List[int] = []
    for d in dirs:
        try:
            with os.scandir(os.path.join(captures_dir, d)) as it:
                for entry in it:
                    if entry.name.endswith(".jpg"):
                        sizes.append(entry.stat().st_size)
                        if len(sizes) >= samples:
                            return statistics.mean(sizes)
        except OSError:
            continue
    return statistics.mean(sizes) if sizes else 120 * 1024


def cmd_plan(args: argparse.Namespace) -> None:
    captures_dir = os.path.abspath(args.captures_dir)
    if not os.path.isdir(captures_dir):
        raise SystemExit(f"captures dir not found: {captures_dir}")

    print(f"scanning {captures_dir} ...")
    pano_dirs = sorted(
        entry.name
        for entry in os.scandir(captures_dir)
        if entry.is_dir(follow_symlinks=False)
    )
    print(f"pano_dirs={len(pano_dirs)}")

    mean_bytes = _sample_mean_file_bytes(captures_dir, pano_dirs[:50])
    print(f"sampled mean file size: {mean_bytes / 1024:.0f} KiB")

    shard_budget = int(args.shard_gb * (1024**3))
    shards: List[dict] = []
    current_dirs: List[str] = []
    current_files = 0
    current_bytes = 0
    total_files = 0
    started = time.time()

    def close_shard():
        nonlocal current_dirs, current_files, current_bytes
        if not current_dirs:
            return
        shards.append(
            {
                "id": f"shard_{len(shards):04d}",
                "dirs": current_dirs,
                "files": current_files,
                "est_bytes": current_bytes,
                "status": "pending",
            }
        )
        current_dirs, current_files, current_bytes = [], 0, 0

    for i, d in enumerate(pano_dirs):
        try:
            with os.scandir(os.path.join(captures_dir, d)) as it:
                n_files = sum(1 for e in it if e.name.endswith(".jpg"))
        except OSError:
            n_files = 0
        if n_files == 0:
            continue
        est = int(n_files * mean_bytes)
        if current_bytes + est > shard_budget and current_dirs:
            close_shard()
        current_dirs.append(d)
        current_files += n_files
        current_bytes += est
        total_files += n_files
        if (i + 1) % 5000 == 0:
            print(f"  scanned {i + 1}/{len(pano_dirs)} dirs ({time.time() - started:.0f}s)")
    close_shard()

    manifest = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "captures_dir": captures_dir,
        "volume_name": VOLUME_NAME,
        "environment": args.environment,
        "shard_gb": args.shard_gb,
        "mean_file_bytes": mean_bytes,
        "total_dirs": sum(len(s["dirs"]) for s in shards),
        "total_files": total_files,
        "total_est_bytes": sum(s["est_bytes"] for s in shards),
        "shards": shards,
    }
    _save_manifest(args.manifest, manifest)

    est_gb = manifest["total_est_bytes"] / (1024**3)
    print(
        f"\nplan written to {args.manifest}\n"
        f"  shards={len(shards)} (~{args.shard_gb} GiB each)\n"
        f"  dirs={manifest['total_dirs']} files={total_files}\n"
        f"  estimated total: {est_gb:.0f} GiB\n"
        f"  volume: {VOLUME_NAME} (env: {args.environment})\n"
        f"\nnothing uploaded. next: python -m utils.modal_volume_upload upload --limit 1"
    )


def _make_tar(captures_dir: str, dirs: List[str], tar_path: str) -> None:
    with tempfile.NamedTemporaryFile("w", suffix=".list", delete=False) as fh:
        fh.write("\n".join(dirs) + "\n")
        list_path = fh.name
    try:
        subprocess.run(
            ["tar", "-cf", tar_path, "-C", captures_dir, "-T", list_path],
            check=True,
        )
    finally:
        os.unlink(list_path)


def cmd_upload(args: argparse.Namespace) -> None:
    manifest = _load_manifest(args.manifest)
    pending = [s for s in manifest["shards"] if s["status"] != "done"]
    print(
        f"shards total={len(manifest['shards'])} pending={len(pending)} "
        f"volume={manifest['volume_name']} env={args.environment}"
    )
    if not pending:
        print("nothing to do")
        return
    if args.limit > 0:
        pending = pending[: args.limit]

    tar_dir = args.tar_dir or os.path.dirname(os.path.abspath(args.manifest))
    os.makedirs(tar_dir, exist_ok=True)

    failed: List[str] = []

    def run_shard(shard: dict) -> bool:
        for attempt in range(args.retries + 1):
            try:
                _process_shard(args, manifest, shard, tar_dir)
                return True
            except Exception as exc:
                print(f"[{shard['id']}] attempt {attempt + 1} failed: {exc}", flush=True)
        return False

    with app.run(environment_name=args.environment):
        with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as pool:
            futures = {pool.submit(run_shard, shard): shard["id"] for shard in pending}
            for future in as_completed(futures):
                if not future.result():
                    failed.append(futures[future])

    done = sum(1 for s in manifest["shards"] if s["status"] == "done")
    print(f"\nupload session finished: {done}/{len(manifest['shards'])} shards done")
    if failed:
        print(f"failed shards (re-run to retry): {sorted(failed)}")
        raise SystemExit(1)


def _process_shard(args: argparse.Namespace, manifest: dict, shard: dict, tar_dir: str) -> None:
    captures_dir = manifest["captures_dir"]
    sid = shard["id"]
    tar_name = f"{sid}.tar"
    tar_path = os.path.join(tar_dir, tar_name)

    if shard["status"] != "uploaded":
        est_gb = shard["est_bytes"] / (1024**3)
        print(f"[{sid}] tarring {len(shard['dirs'])} dirs (~{est_gb:.1f} GiB) ...", flush=True)
        t0 = time.time()
        _make_tar(captures_dir, shard["dirs"], tar_path)
        actual_gb = os.path.getsize(tar_path) / (1024**3)
        print(f"[{sid}] tarred {actual_gb:.1f} GiB in {time.time() - t0:.0f}s; uploading ...", flush=True)

        t0 = time.time()
        vol = modal.Volume.from_name(
            manifest["volume_name"],
            create_if_missing=True,
            environment_name=args.environment,
        )
        with vol.batch_upload(force=True) as batch:
            batch.put_file(tar_path, f"/{STAGING_DIR}/{tar_name}")
        upload_s = time.time() - t0
        shard["status"] = "uploaded"
        _save_manifest(args.manifest, manifest)
        print(
            f"[{sid}] uploaded in {upload_s:.0f}s "
            f"({actual_gb * 8 * 1024 / max(upload_s, 1e-9):.0f} Mbps); extracting ...",
            flush=True,
        )
    else:
        print(f"[{sid}] already uploaded; extracting ...", flush=True)

    result = extract_shard.remote(tar_name)
    if os.path.exists(tar_path):
        os.remove(tar_path)
    shard["status"] = "done"
    shard["extracted_files"] = result["extracted"]
    _save_manifest(args.manifest, manifest)
    print(f"[{sid}] done: extracted {result['extracted']} entries", flush=True)


def cmd_status(args: argparse.Namespace) -> None:
    manifest = _load_manifest(args.manifest)
    shards = manifest["shards"]
    done = [s for s in shards if s["status"] == "done"]
    done_bytes = sum(s["est_bytes"] for s in done)
    total_bytes = manifest["total_est_bytes"]
    print(
        f"volume={manifest['volume_name']} shards={len(done)}/{len(shards)} done "
        f"(~{done_bytes / 1024**3:.0f}/{total_bytes / 1024**3:.0f} GiB)"
    )


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--environment", default=DEFAULT_ENVIRONMENT)
    sub = parser.add_subparsers(dest="command", required=True)

    p_plan = sub.add_parser("plan", help="Scan captures and write a shard manifest (local only).")
    p_plan.add_argument("--captures-dir", default="captures")
    p_plan.add_argument("--shard-gb", type=float, default=8.0)
    p_plan.set_defaults(func=cmd_plan)

    p_up = sub.add_parser("upload", help="Tar, upload, and extract pending shards.")
    p_up.add_argument("--limit", type=int, default=0, help="Max shards this session (0 = all).")
    p_up.add_argument("--retries", type=int, default=2, help="Retries per shard before moving on.")
    p_up.add_argument("--concurrency", type=int, default=3, help="Concurrent shard pipelines.")
    p_up.add_argument("--tar-dir", default=None, help="Where to write temp tars (default: manifest dir).")
    p_up.set_defaults(func=cmd_upload)

    p_st = sub.add_parser("status", help="Show manifest progress.")
    p_st.set_defaults(func=cmd_status)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main(sys.argv[1:])
