"""Run partial eval mining + query-adapter training in one loop."""

import argparse
import os
import subprocess
import sys
from datetime import datetime


def _run(cmd):
    print("running:", " ".join(cmd))
    completed = subprocess.run(cmd)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def main():
    parser = argparse.ArgumentParser(
        description="Overnight loop for hard-negative mining and query-adapter fine-tuning."
    )
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--sample-size", type=int, default=1500)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--db-max-top-k", type=int, default=5000)
    parser.add_argument("--ivfflat-probes", type=int, default=120)
    parser.add_argument("--locate-top-k-per-crop", type=int, default=220)
    parser.add_argument("--locate-max-candidates", type=int, default=5000)
    parser.add_argument("--locate-verify-top-n", type=int, default=120)
    parser.add_argument("--locate-min-good-matches", type=int, default=10)
    parser.add_argument("--locate-min-inlier-ratio", type=float, default=0.14)
    parser.add_argument("--train-model-id", default="clip")
    parser.add_argument("--train-max-triplets", type=int, default=80000)
    parser.add_argument("--train-epochs", type=int, default=4)
    parser.add_argument("--train-batch-size", type=int, default=128)
    parser.add_argument("--train-margin", type=float, default=0.2)
    parser.add_argument("--train-lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="artifacts/hard_negative_loop")
    args = parser.parse_args()

    root_dir = os.path.abspath(args.output_dir)
    os.makedirs(root_dir, exist_ok=True)
    now = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(root_dir, now)
    os.makedirs(run_dir, exist_ok=True)

    print(f"run_dir={run_dir}")
    last_adapter = ""

    for i in range(1, max(1, int(args.iterations)) + 1):
        iteration_tag = f"iter_{i:02d}"
        iter_dir = os.path.join(run_dir, iteration_tag)
        os.makedirs(iter_dir, exist_ok=True)

        results_csv = os.path.join(iter_dir, "partial_eval_results.csv")
        hard_csv = os.path.join(iter_dir, "partial_eval_hard_negatives.csv")
        adapter_path = os.path.join(
            iter_dir, f"query_adapter_{str(args.train_model_id).lower()}.pt"
        )

        eval_cmd = [
            sys.executable,
            "-m",
            "utils.eval_retrieval_partials",
            "--sample-size",
            str(args.sample_size),
            "--mode",
            "both",
            "--top-k",
            str(args.top_k),
            "--db-max-top-k",
            str(args.db_max_top_k),
            "--ivfflat-probes",
            str(args.ivfflat_probes),
            "--locate-top-k-per-crop",
            str(args.locate_top_k_per_crop),
            "--locate-max-candidates",
            str(args.locate_max_candidates),
            "--locate-verify-top-n",
            str(args.locate_verify_top_n),
            "--locate-min-good-matches",
            str(args.locate_min_good_matches),
            "--locate-min-inlier-ratio",
            str(args.locate_min_inlier_ratio),
            "--out-csv",
            results_csv,
            "--hard-negatives-csv",
            hard_csv,
        ]
        _run(eval_cmd)

        train_cmd = [
            sys.executable,
            "-m",
            "utils.train_retrieval_query_adapter",
            "--hard-negatives-csv",
            hard_csv,
            "--output",
            adapter_path,
            "--model-id",
            str(args.train_model_id),
            "--mode",
            "both",
            "--max-triplets",
            str(args.train_max_triplets),
            "--epochs",
            str(args.train_epochs),
            "--batch-size",
            str(args.train_batch_size),
            "--margin",
            str(args.train_margin),
            "--lr",
            str(args.train_lr),
            "--seed",
            str(args.seed),
        ]
        _run(train_cmd)
        last_adapter = adapter_path
        print(f"{iteration_tag}_adapter={adapter_path}")

    if last_adapter:
        print(f"latest_adapter={last_adapter}")
        print(
            "set_env="
            f"GEOSPY_RETRIEVAL_QUERY_ADAPTER_{str(args.train_model_id).upper()}_PATH={last_adapter}"
        )


if __name__ == "__main__":
    main()

