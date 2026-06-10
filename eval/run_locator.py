"""Run automated locator evaluation against the backend API."""

import argparse
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional, Sequence, Tuple

from eval.common import EvalCase, haversine_m, load_eval_cases, write_csv, write_json
from eval.http_client import maybe_str, post_image_for_json
from eval.metrics import (
    compare_to_baseline,
    summarize_by_category,
    summarize_negative,
    summarize_positive,
)


@dataclass
class LocateSettings:
    settings_id: str = "default"
    top_k: int = 8
    min_similarity: Optional[float] = None
    embedding_base: str = ""
    orb_enabled: Optional[bool] = None
    orb_top_n: Optional[int] = None
    orb_weight: Optional[float] = None
    orb_feature_count: Optional[int] = None
    orb_ransac_top_k: Optional[int] = None
    orb_ignore_bottom_ratio: Optional[float] = None
    sam2_mask_cars: Optional[bool] = None
    sam2_mask_trees: Optional[bool] = None


def _prediction_lat_lon(match: dict) -> Tuple[Optional[float], Optional[float]]:
    lat = match.get("family_center_lat", match.get("lat"))
    lon = match.get("family_center_lon", match.get("lon"))
    try:
        if lat is None or lon is None:
            return None, None
        return float(lat), float(lon)
    except Exception:
        return None, None


def _evaluate_case(
    *,
    case: EvalCase,
    endpoint_url: str,
    settings: LocateSettings,
    timeout_seconds: float,
) -> dict:
    if not os.path.exists(case.image_path):
        row = {
            "case_id": case.case_id,
            "settings_id": settings.settings_id,
            "status": "missing-image",
            "image_path": case.image_path,
            "expected_reject": int(case.expected_reject),
        }
        row.update(_settings_to_row(settings))
        return row
    fields = _settings_to_form_fields(settings)
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
    top_panorama_id = _optional_int(top_match.get("panorama_id"))
    top_capture_id = _optional_int(top_match.get("capture_id"))
    expected_panorama_rank = _rank_match(
        matches,
        field="panorama_id",
        expected=case.expected_panorama_id,
    )
    expected_capture_rank = _rank_match(
        matches,
        field="capture_id",
        expected=case.expected_capture_id,
    )
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
        "settings_id": settings.settings_id,
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
        "top_panorama_id": top_panorama_id,
        "top_capture_id": top_capture_id,
        "top_family_score": top_match.get("family_score"),
        "expected_panorama_rank": expected_panorama_rank,
        "expected_capture_rank": expected_capture_rank,
        "panorama_top1": (
            int(top_panorama_id == int(case.expected_panorama_id))
            if case.expected_panorama_id is not None
            else None
        ),
        "capture_top1": (
            int(top_capture_id == int(case.expected_capture_id))
            if case.expected_capture_id is not None
            else None
        ),
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
    result.update(_settings_to_row(settings))
    return result


def _optional_int(value: object) -> Optional[int]:
    try:
        if value is None or str(value).strip() == "":
            return None
        return int(value)
    except Exception:
        return None


def _rank_match(matches: Sequence[dict], *, field: str, expected: Optional[int]) -> Optional[int]:
    if expected is None:
        return None
    expected_int = int(expected)
    for idx, match in enumerate(matches, start=1):
        if _optional_int(match.get(field)) == expected_int:
            return idx
    return None


def _parse_boolish(value: object) -> Optional[bool]:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    raw = str(value).strip().lower()
    if raw in {"1", "true", "yes", "y", "on"}:
        return True
    if raw in {"0", "false", "no", "n", "off"}:
        return False
    return None


def _slugify(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value or "").strip()).strip("-")
    return slug[:80] or "settings"


def _settings_to_row(settings: LocateSettings) -> dict:
    row = asdict(settings)
    return {f"setting_{key}": value for key, value in row.items()}


def _settings_to_form_fields(settings: LocateSettings) -> Dict[str, str]:
    fields: Dict[str, str] = {"top_k": str(max(1, int(settings.top_k)))}
    optional_values = {
        "min_similarity": settings.min_similarity,
        "embedding_base": settings.embedding_base,
        "orb_top_n": settings.orb_top_n,
        "orb_weight": settings.orb_weight,
        "orb_feature_count": settings.orb_feature_count,
        "orb_ransac_top_k": settings.orb_ransac_top_k,
        "orb_ignore_bottom_ratio": settings.orb_ignore_bottom_ratio,
    }
    for key, value in optional_values.items():
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        fields[key] = str(value)
    bool_values = {
        "orb_enabled": settings.orb_enabled,
        "sam2_mask_cars": settings.sam2_mask_cars,
        "sam2_mask_trees": settings.sam2_mask_trees,
    }
    for key, value in bool_values.items():
        if value is None:
            continue
        fields[key] = "1" if bool(value) else "0"
    return fields


def _settings_from_mapping(mapping: dict, fallback_id: str) -> LocateSettings:
    def pick(*names: str, default=None):
        for name in names:
            if name in mapping:
                return mapping[name]
        return default

    settings_id = str(pick("settings_id", "id", "name", default=fallback_id) or fallback_id)
    return LocateSettings(
        settings_id=settings_id,
        top_k=max(1, int(pick("top_k", "topK", default=8) or 8)),
        min_similarity=_optional_float(pick("min_similarity", "minSimilarity")),
        embedding_base=str(pick("embedding_base", "embeddingBase", default="") or "").strip(),
        orb_enabled=_parse_boolish(pick("orb_enabled", "orbEnabled")),
        orb_top_n=_optional_int(pick("orb_top_n", "orbTopN")),
        orb_weight=_optional_float(pick("orb_weight", "orbWeight")),
        orb_feature_count=_optional_int(pick("orb_feature_count", "orbFeatureCount")),
        orb_ransac_top_k=_optional_int(pick("orb_ransac_top_k", "orbRansacTopK")),
        orb_ignore_bottom_ratio=_optional_float(
            pick("orb_ignore_bottom_ratio", "orbIgnoreBottomRatio")
        ),
        sam2_mask_cars=_parse_boolish(pick("sam2_mask_cars", "sam2MaskCars")),
        sam2_mask_trees=_parse_boolish(pick("sam2_mask_trees", "sam2MaskTrees")),
    )


def _optional_float(value: object) -> Optional[float]:
    try:
        if value is None or str(value).strip() == "":
            return None
        return float(value)
    except Exception:
        return None


def _settings_from_json_payload(payload: object) -> List[LocateSettings]:
    if isinstance(payload, dict) and isinstance(payload.get("settings"), list):
        payload = payload["settings"]
    if isinstance(payload, dict):
        payload = [payload]
    if not isinstance(payload, list):
        raise ValueError("settings JSON must be an object, a list, or {'settings': [...]}")
    settings: List[LocateSettings] = []
    for idx, item in enumerate(payload, start=1):
        if not isinstance(item, dict):
            raise ValueError("each settings entry must be an object")
        settings.append(_settings_from_mapping(item, fallback_id=f"settings-{idx}"))
    return settings


def _load_settings(args: argparse.Namespace) -> List[LocateSettings]:
    json_payload = None
    if args.settings_json:
        with open(os.path.abspath(args.settings_json), "r", encoding="utf-8") as f:
            json_payload = json.load(f)
    elif args.settings_json_inline:
        json_payload = json.loads(args.settings_json_inline)

    if json_payload is not None:
        settings = _settings_from_json_payload(json_payload)
        if not settings:
            raise SystemExit("Settings JSON did not contain any settings entries.")
        return settings

    return [
        LocateSettings(
            settings_id=str(args.settings_id or "default"),
            top_k=max(1, int(args.top_k)),
            min_similarity=args.min_similarity,
            embedding_base=str(args.embedding_base or "").strip(),
            orb_enabled=args.orb_enabled,
            orb_top_n=args.orb_top_n,
            orb_weight=args.orb_weight,
            orb_feature_count=args.orb_feature_count,
            orb_ransac_top_k=args.orb_ransac_top_k,
            orb_ignore_bottom_ratio=args.orb_ignore_bottom_ratio,
            sam2_mask_cars=args.sam2_mask_cars,
            sam2_mask_trees=args.sam2_mask_trees,
        )
    ]


def _evaluate_settings(
    *,
    cases: Sequence[EvalCase],
    settings: LocateSettings,
    endpoint_url: str,
    timeout_seconds: float,
    concurrency: int,
) -> List[dict]:
    result_rows: List[dict] = []
    max_workers = max(1, int(concurrency))
    if max_workers <= 1:
        for idx, case in enumerate(cases, start=1):
            row = _evaluate_case(
                case=case,
                endpoint_url=endpoint_url,
                settings=settings,
                timeout_seconds=timeout_seconds,
            )
            result_rows.append(row)
            _print_case_progress(idx, len(cases), row)
        return result_rows

    rows_by_index: Dict[int, dict] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _evaluate_case,
                case=case,
                endpoint_url=endpoint_url,
                settings=settings,
                timeout_seconds=timeout_seconds,
            ): idx
            for idx, case in enumerate(cases, start=1)
        }
        completed = 0
        for future in as_completed(futures):
            idx = futures[future]
            row = future.result()
            completed += 1
            rows_by_index[idx] = row
            _print_case_progress(completed, len(cases), row)
    for idx in sorted(rows_by_index):
        result_rows.append(rows_by_index[idx])
    return result_rows


def _flat_summary(summary: dict) -> dict:
    """Summary with nested values (CIs, category breakdowns) removed for CSV output."""
    return {key: value for key, value in summary.items() if not isinstance(value, dict)}


def _print_case_progress(idx: int, total: int, row: dict) -> None:
    print(
        f"[{idx}/{total}] settings_id={row.get('settings_id')} "
        f"case_id={row.get('case_id')} status={row.get('status')} "
        f"matches={row.get('match_count', 0)} error_m={row.get('error_m')}"
    )


def _summarize_rows(
    *,
    settings: LocateSettings,
    endpoint_url: str,
    rows: List[dict],
    args: argparse.Namespace,
) -> dict:
    positive_rows = [row for row in rows if not int(row.get("expected_reject", 0))]
    negative_rows = [row for row in rows if int(row.get("expected_reject", 0))]
    ok_count = sum(1 for row in rows if row.get("status") == "ok")
    elapsed_values = [float(row.get("elapsed_ms") or 0.0) for row in rows]
    mean_elapsed = (
        round(sum(elapsed_values) / float(len(elapsed_values)), 2)
        if elapsed_values
        else 0.0
    )
    summary = {
        "settings_id": settings.settings_id,
        "endpoint": endpoint_url,
        "limit": int(args.limit),
        "concurrency": max(1, int(args.concurrency)),
        "reject_family_score_threshold": args.reject_family_score_threshold,
        "total_cases": len(rows),
        "ok_cases": ok_count,
        "ok_rate": round(100.0 * ok_count / len(rows), 2) if rows else 0.0,
        "mean_elapsed_ms": mean_elapsed,
        **_settings_to_row(settings),
        **summarize_positive(positive_rows, seed=int(args.seed)),
        **summarize_negative(negative_rows, args.reject_family_score_threshold),
        "categories": summarize_by_category(
            rows,
            args.reject_family_score_threshold,
            seed=int(args.seed),
        ),
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run automated locate evals.")
    parser.add_argument("--cases", default="eval/datasets/locator_cases.csv")
    parser.add_argument(
        "--endpoint",
        default="http://127.0.0.1:8000/api/retrieval/locate-by-image",
    )
    parser.add_argument("--settings-id", default="default")
    parser.add_argument("--settings-json", default="", help="Path to a JSON object/list of locate settings.")
    parser.add_argument(
        "--settings-json-inline",
        default="",
        help="Inline JSON object/list of locate settings. Useful for API-launched evals.",
    )
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--embedding-base", default="")
    parser.add_argument("--min-similarity", type=float, default=None)
    parser.add_argument("--orb-enabled", dest="orb_enabled", action="store_true", default=None)
    parser.add_argument("--orb-disabled", dest="orb_enabled", action="store_false")
    parser.add_argument("--orb-top-n", type=int, default=None)
    parser.add_argument("--orb-weight", type=float, default=None)
    parser.add_argument("--orb-feature-count", type=int, default=None)
    parser.add_argument("--orb-ransac-top-k", type=int, default=None)
    parser.add_argument("--orb-ignore-bottom-ratio", type=float, default=None)
    parser.add_argument("--sam2-mask-cars", dest="sam2_mask_cars", action="store_true", default=None)
    parser.add_argument("--no-sam2-mask-cars", dest="sam2_mask_cars", action="store_false")
    parser.add_argument("--sam2-mask-trees", dest="sam2_mask_trees", action="store_true", default=None)
    parser.add_argument("--no-sam2-mask-trees", dest="sam2_mask_trees", action="store_false")
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument(
        "--reject-family-score-threshold",
        type=float,
        default=None,
        help="Optional threshold for counting low-score matches as correct rejects on negative cases.",
    )
    parser.add_argument("--output-dir", default="")
    parser.add_argument(
        "--baseline",
        default="",
        help="Path to a baseline summary.json to compare this run against.",
    )
    parser.add_argument(
        "--save-baseline",
        default="",
        help="Path to write this run's summary as the new regression baseline "
        "(e.g. eval/baselines/locator_baseline.json).",
    )
    args = parser.parse_args()

    cases = load_eval_cases(os.path.abspath(args.cases))
    if not cases:
        raise SystemExit("No eval cases loaded.")
    if args.shuffle:
        import random

        random.Random(int(args.seed)).shuffle(cases)
    if args.limit and int(args.limit) > 0:
        cases = cases[: int(args.limit)]
    settings_list = _load_settings(args)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.abspath(
        args.output_dir or os.path.join("eval", "results", f"locator_eval_{timestamp}")
    )
    os.makedirs(output_dir, exist_ok=True)

    all_rows: List[dict] = []
    summaries: List[dict] = []
    multi_settings = len(settings_list) > 1
    for settings_idx, settings in enumerate(settings_list, start=1):
        print(f"settings_start={settings.settings_id} index={settings_idx}/{len(settings_list)}")
        rows = _evaluate_settings(
            cases=cases,
            settings=settings,
            endpoint_url=str(args.endpoint),
            timeout_seconds=max(1.0, float(args.timeout_seconds)),
            concurrency=max(1, int(args.concurrency)),
        )
        summary = _summarize_rows(
            settings=settings,
            endpoint_url=str(args.endpoint),
            rows=rows,
            args=args,
        )
        summaries.append(summary)
        all_rows.extend(rows)

        setting_output_dir = output_dir
        if multi_settings:
            setting_output_dir = os.path.join(
                output_dir,
                "settings",
                f"{settings_idx:02d}_{_slugify(settings.settings_id)}",
            )
            os.makedirs(setting_output_dir, exist_ok=True)
        write_csv(os.path.join(setting_output_dir, "case_results.csv"), rows)
        write_json(os.path.join(setting_output_dir, "summary.json"), summary)
        print(
            f"settings_done={settings.settings_id} "
            f"within_50m={summary.get('within_50m')} median_error_m={summary.get('median_error_m')}"
        )

    write_csv(os.path.join(output_dir, "all_case_results.csv"), all_rows)
    write_csv(
        os.path.join(output_dir, "combined_summary.csv"),
        [_flat_summary(summary) for summary in summaries],
    )
    write_json(
        os.path.join(output_dir, "combined_summary.json"),
        {
            "cases": os.path.abspath(args.cases),
            "endpoint": str(args.endpoint),
            "settings_count": len(settings_list),
            "case_count": len(cases),
            "summaries": summaries,
        },
    )
    print(f"output_dir={output_dir}")
    if len(summaries) == 1:
        primary = summaries[0]
        for key, value in _flat_summary(primary).items():
            print(f"{key}={value}")
        for category, stats in (primary.get("categories") or {}).items():
            print(f"category={category} " + " ".join(
                f"{key}={value}"
                for key, value in stats.items()
                if not isinstance(value, dict)
            ))
    else:
        best = max(summaries, key=lambda item: (float(item.get("within_50m") or 0.0), -float(item.get("median_error_m") or 1e18)))
        primary = best
        print(f"best_settings_id={best.get('settings_id')}")
        print(f"best_within_50m={best.get('within_50m')}")
        print(f"best_median_error_m={best.get('median_error_m')}")

    if args.baseline:
        baseline_path = os.path.abspath(args.baseline)
        with open(baseline_path, "r", encoding="utf-8") as f:
            baseline = json.load(f)
        comparison = compare_to_baseline(primary, baseline)
        write_json(os.path.join(output_dir, "baseline_comparison.json"), comparison)
        regressions = [
            metric
            for metric, item in comparison.items()
            if item.get("improved") is False
        ]
        print(f"baseline={baseline_path}")
        for metric, item in comparison.items():
            print(
                f"baseline_delta {metric}: {item['baseline']} -> {item['current']} "
                f"(delta={item['delta']:+})"
            )
        if regressions:
            print(f"baseline_regressions={','.join(regressions)}")
        else:
            print("baseline_regressions=none")

    if args.save_baseline:
        baseline_out = os.path.abspath(args.save_baseline)
        os.makedirs(os.path.dirname(baseline_out), exist_ok=True)
        write_json(baseline_out, primary)
        print(f"saved_baseline={baseline_out}")


if __name__ == "__main__":
    main()
