"""Run automated locator evaluation against the backend API."""

import argparse
import os
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from eval.common import EvalCase, haversine_m, load_eval_cases, write_csv, write_json
from eval.http_client import maybe_str, post_image_for_json


def _prediction_lat_lon(match: dict) -> Tuple[Optional[float], Optional[float]]:
    lat = match.get("family_center_lat", match.get("lat"))
    lon = match.get("family_center_lon", match.get("lon"))
    try:
        if lat is None or lon is None:
            return None, None
        return float(lat), float(lon)
    except Exception:
        return None, None


def _summarize_positive(rows: List[dict]) -> Dict[str, float]:
    total = len(rows)
    if total <= 0:
        return {
            "positive_cases": 0,
            "within_25m": 0.0,
            "within_50m": 0.0,
            "within_100m": 0.0,
            "median_error_m": 0.0,
        }
    errors = [float(row["error_m"]) for row in rows if row.get("error_m") is not None]
    errors.sort()
    def pct(threshold: float) -> float:
        return round(
            100.0 * sum(1 for row in rows if float(row.get("error_m") or 1e18) <= threshold) / total,
            2,
        )
    median_error = errors[len(errors) // 2] if errors else 0.0
    return {
        "positive_cases": total,
        "within_25m": pct(25.0),
        "within_50m": pct(50.0),
        "within_100m": pct(100.0),
        "median_error_m": round(float(median_error), 2),
    }


def _summarize_negative(
    rows: List[dict], reject_family_score_threshold: Optional[float]
) -> Dict[str, float]:
    total = len(rows)
    if total <= 0:
        return {
            "negative_cases": 0,
            "reject_rate": 0.0,
            "false_accept_rate": 0.0,
        }
    correct_rejects = 0
    for row in rows:
        has_match = bool(row.get("top_family_id"))
        top_score = row.get("top_family_score")
        rejected = not has_match
        if (
            not rejected
            and reject_family_score_threshold is not None
            and top_score is not None
            and float(top_score) < float(reject_family_score_threshold)
        ):
            rejected = True
        if rejected:
            correct_rejects += 1
    reject_rate = round(100.0 * correct_rejects / total, 2)
    return {
        "negative_cases": total,
        "reject_rate": reject_rate,
        "false_accept_rate": round(100.0 - reject_rate, 2),
    }


def _evaluate_case(
    *,
    case: EvalCase,
    endpoint_url: str,
    top_k: int,
    min_similarity: Optional[float],
    timeout_seconds: float,
) -> dict:
    if not os.path.exists(case.image_path):
        return {
            "case_id": case.case_id,
            "status": "missing-image",
            "image_path": case.image_path,
            "expected_reject": int(case.expected_reject),
        }
    fields = {"top_k": str(max(1, int(top_k)))}
    if min_similarity is not None:
        fields["min_similarity"] = str(float(min_similarity))
    started = time.perf_counter()
    status_code, payload = post_image_for_json(
        url=endpoint_url,
        file_path=case.image_path,
        fields=fields,
        timeout_seconds=timeout_seconds,
    )
    elapsed_ms = round((time.perf_counter() - started) * 1000.0, 2)
    matches = list(payload.get("matches") or []) if isinstance(payload, dict) else []
    top_match = matches[0] if matches else {}
    pred_lat, pred_lon = _prediction_lat_lon(top_match)
    error_m = None
    if (
        not case.expected_reject
        and pred_lat is not None
        and pred_lon is not None
        and case.expected_lat is not None
        and case.expected_lon is not None
    ):
        error_m = round(haversine_m(case.expected_lat, case.expected_lon, pred_lat, pred_lon), 2)
    result = {
        "case_id": case.case_id,
        "status": "ok" if status_code == 200 else f"http-{status_code}",
        "http_status": status_code,
        "elapsed_ms": elapsed_ms,
        "image_path": case.image_path,
        "expected_reject": int(case.expected_reject),
        "expected_lat": case.expected_lat,
        "expected_lon": case.expected_lon,
        "expected_panorama_id": case.expected_panorama_id,
        "expected_capture_id": case.expected_capture_id,
        "split": case.split,
        "notes": case.notes,
        "retrieval_id": maybe_str(payload.get("retrieval_id") if isinstance(payload, dict) else ""),
        "capture_candidates": payload.get("capture_candidates") if isinstance(payload, dict) else None,
        "panorama_candidates": payload.get("panorama_candidates") if isinstance(payload, dict) else None,
        "match_count": len(matches),
        "top_family_id": maybe_str(top_match.get("family_id")),
        "top_panorama_id": top_match.get("panorama_id"),
        "top_family_score": top_match.get("family_score"),
        "pred_lat": pred_lat,
        "pred_lon": pred_lon,
        "error_m": error_m,
        "within_25m": int(error_m is not None and error_m <= 25.0),
        "within_50m": int(error_m is not None and error_m <= 50.0),
        "within_100m": int(error_m is not None and error_m <= 100.0),
        "backend_detail": (
            payload.get("detail", "") if isinstance(payload, dict) else ""
        ),
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run automated locate evals.")
    parser.add_argument("--cases", default="eval/datasets/locator_cases.csv")
    parser.add_argument(
        "--endpoint",
        default="http://127.0.0.1:8000/api/retrieval/locate-by-image",
    )
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--min-similarity", type=float, default=None)
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--reject-family-score-threshold",
        type=float,
        default=None,
        help="Optional threshold for counting low-score matches as correct rejects on negative cases.",
    )
    parser.add_argument("--output-dir", default="")
    args = parser.parse_args()

    cases = load_eval_cases(os.path.abspath(args.cases))
    if not cases:
        raise SystemExit("No eval cases loaded.")
    if args.limit and int(args.limit) > 0:
        cases = cases[: int(args.limit)]

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.abspath(
        args.output_dir or os.path.join("eval", "results", f"locator_eval_{timestamp}")
    )
    os.makedirs(output_dir, exist_ok=True)

    result_rows: List[dict] = []
    for idx, case in enumerate(cases, start=1):
        row = _evaluate_case(
            case=case,
            endpoint_url=str(args.endpoint),
            top_k=max(1, int(args.top_k)),
            min_similarity=args.min_similarity,
            timeout_seconds=max(1.0, float(args.timeout_seconds)),
        )
        result_rows.append(row)
        print(
            f"[{idx}/{len(cases)}] case_id={case.case_id} status={row['status']} "
            f"matches={row.get('match_count', 0)} error_m={row.get('error_m')}"
        )

    positive_rows = [row for row in result_rows if not int(row.get("expected_reject", 0))]
    negative_rows = [row for row in result_rows if int(row.get("expected_reject", 0))]
    summary = {
        "endpoint": str(args.endpoint),
        "top_k": int(args.top_k),
        "min_similarity": args.min_similarity,
        "limit": int(args.limit),
        "reject_family_score_threshold": args.reject_family_score_threshold,
        "total_cases": len(result_rows),
        **_summarize_positive(positive_rows),
        **_summarize_negative(negative_rows, args.reject_family_score_threshold),
    }

    write_csv(os.path.join(output_dir, "case_results.csv"), result_rows)
    write_json(os.path.join(output_dir, "summary.json"), summary)
    print(f"output_dir={output_dir}")
    for key, value in summary.items():
        print(f"{key}={value}")


if __name__ == "__main__":
    main()
