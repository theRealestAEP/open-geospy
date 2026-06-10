# Eval Framework

This folder is the start of a reusable evaluation framework for retrieval and locator experiments.

## Leakage Warning — read this first

`build_locator_dataset.py` samples its positives from the **indexed captures
themselves**, so any run against that manifest is a smoke test, not a real
accuracy measurement (the system is shown images it has literally indexed).

For honest numbers, build a held-out set with `scrape_locator_dataset.py`:
it fetches fresh Street View renders (different pixels from the indexed
captures) and records whether each pano exists in the index. The runner then
reports metrics per category:

- `in_index_pano` — query pano exists in the index. Tests recognition of a
  known panorama from a new render/heading/pitch.
- `novel_pano` — query pano is NOT in the index but has ground-truth coords.
  Tests spatial generalization: the system should still land within 25–100m
  via neighboring panoramas.
- `negative` — `expected_reject=1` cases the system should refuse.

## Files

- `build_locator_dataset.py`
  - Samples positive cases from the local DB and optionally appends manual negative images from a folder.
- `run_locator.py`
  - Calls `/api/retrieval/locate-by-image` and scores location accuracy automatically.
  - Can run one settings profile or a JSON settings matrix.
- `scrape_locator_dataset.py`
  - Samples random Street View images inside a bbox/polygon and writes a locator manifest.
- `common.py`
  - Shared case loading, CSV writing, and distance math.
- `http_client.py`
  - Minimal multipart HTTP client so evals can hit the running backend without extra dependencies.

## Case Manifest

The locator eval reads a CSV with these columns:

```csv
case_id,image_path,expected_lat,expected_lon,expected_panorama_id,expected_capture_id,expected_reject,split,notes
```

Rules:

- `image_path` can be absolute or relative to the CSV file.
- Positive cases set `expected_reject=0` and include the expected lat/lon.
- Negative cases set `expected_reject=1` and usually leave lat/lon blank.

## Quick Start

Build a starter manifest from local captures:

```bash
python -m eval.build_locator_dataset \
  --output eval/datasets/locator_cases.csv \
  --positive-count 200
```

Append manual negatives from a folder:

```bash
python -m eval.build_locator_dataset \
  --output eval/datasets/locator_cases.csv \
  --positive-count 200 \
  --negative-dir eval/datasets/manual_negatives
```

Run the locator eval against the running backend:

```bash
python -m eval.run_locator \
  --cases eval/datasets/locator_cases.csv \
  --endpoint http://127.0.0.1:8000/api/retrieval/locate-by-image
```

Run a settings matrix:

```bash
python -m eval.run_locator \
  --cases eval/datasets/locator_cases.csv \
  --settings-json eval/settings/locate_matrix.json \
  --concurrency 2
```

Example settings JSON:

```json
[
  {
    "id": "clip-baseline",
    "top_k": 8,
    "embedding_base": "clip",
    "orb_enabled": false
  },
  {
    "id": "clip-orb",
    "top_k": 8,
    "embedding_base": "clip",
    "orb_enabled": true,
    "orb_top_n": 100,
    "orb_weight": 0.75,
    "orb_feature_count": 500,
    "orb_ransac_top_k": 10,
    "orb_ignore_bottom_ratio": 0.28
  }
]
```

Scrape a held-out eval set inside a boundary (aim for 500+ cases so the
within-50m confidence interval is tighter than a few points):

```bash
python -m eval.scrape_locator_dataset \
  --bbox 37.774,-122.431,37.787,-122.412 \
  --count 500 \
  --split held-out \
  --output-dir eval/datasets/held_out
```

Results land under `eval/results/...` with:

- `case_results.csv`
- `summary.json`
- `combined_summary.csv` and `combined_summary.json` for matrix runs
- `all_case_results.csv` for matrix runs

## What It Measures

For positive cases:

- `within_25m`
- `within_50m`
- `within_100m`
- `panorama_top1` when `expected_panorama_id` is present
- `capture_top1` when `expected_capture_id` is present
- `median_error_m`

For negative cases:

- `reject_rate`
- `false_accept_rate`

All rate metrics carry 95% Wilson confidence intervals (`*_ci95`), and
`median_error_m` carries a bootstrap CI. `summary.json` also includes a
`categories` block with the same metrics broken down by
`in_index_pano` / `novel_pano` / `negative`.

## Regression Baselines

Snapshot a run as the regression baseline:

```bash
python -m eval.run_locator \
  --cases eval/datasets/held_out/locator_cases.csv \
  --save-baseline eval/baselines/locator_baseline.json
```

Compare any later run against it:

```bash
python -m eval.run_locator \
  --cases eval/datasets/held_out/locator_cases.csv \
  --baseline eval/baselines/locator_baseline.json
```

The runner prints per-metric deltas, flags `baseline_regressions=...`, and
writes `baseline_comparison.json` into the results directory. Commit
`eval/baselines/` so regressions are visible in review.

## Negatives

Yes, this helps with negatives.

It gives you a place to track:

- out-of-index images you expect the system to reject
- low-information images like blank sky or motion blur
- domain-shift images from other cities or cameras

You can start with a manual folder of negatives now and later add automatic negative harvesting.

## Frontend Runner

The app has an `Eval` tab that launches `python -m eval.run_locator` through the backend, polls job status, and shows the combined summary. The backend writes output under `eval/results/frontend_locator_eval_<id>` unless you provide an output directory.
