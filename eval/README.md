# Eval Framework

This folder is the start of a reusable evaluation framework for retrieval and locator experiments.

## Files

- `build_locator_dataset.py`
  - Samples positive cases from the local DB and optionally appends manual negative images from a folder.
- `run_locator.py`
  - Calls `/api/retrieval/locate-by-image` and scores location accuracy automatically.
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

Results land under `eval/results/...` with:

- `case_results.csv`
- `summary.json`

## What It Measures

For positive cases:

- `within_25m`
- `within_50m`
- `within_100m`
- `median_error_m`

For negative cases:

- `reject_rate`
- `false_accept_rate`

## Negatives

Yes, this helps with negatives.

It gives you a place to track:

- out-of-index images you expect the system to reject
- low-information images like blank sky or motion blur
- domain-shift images from other cities or cameras

You can start with a manual folder of negatives now and later add automatic negative harvesting.
