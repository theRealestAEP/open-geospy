# Retrieval Pipeline Write-Up

This document explains the current search-only retrieval pipeline, why each stage exists, and where the place-recognition model fits in.

## What The Pipeline Does

Given a query image, the system tries to:

1. find similar captures in the DB,
2. combine evidence across crops and models,
3. return a ranked top-K set with similarity scores.

The same embedding stack is used for:

- `search-by-image` (ranked matches),
- background/index backfill pipelines.

## High-Level Flow

```mermaid
flowchart LR
queryImage[QueryImageUpload] --> embedModels[ConfiguredImageEmbedders]
embedModels --> annSearch[PgVectorANNSearch]
annSearch --> fusion[MultiModelScoreFusion]
fusion --> rankedOutput[TopKRankedMatches]
```

## Detailed Stages (And Why)

### 1) Multi-crop query embeddings

The query image is split into multiple crops (full, center, side/top/bottom variants).

**Why**:
- real user inputs are often partial (building segment, storefront corner),
- one global embedding can overfit generic context (road/sky),
- crop diversity increases recall for localized structure.

### 2) Multi-model retrieval

The system runs multiple embedders (primary CLIP + optional place branch), then searches pgvector for each.

**Why**:
- CLIP is strong semantically, but may confuse visually similar streets,
- place-style embeddings are usually better at location-discriminative cues,
- fusing both is more robust than either alone.

### 3) Candidate fusion

All candidate rows are merged by capture id, weighted by model and crop hits.

**Why**:
- fusion stabilizes ranking across noisy crops,
### 4) Final ranked output

After fusion, matches are sorted and returned with `score` / `similarity`.

**Why**:
- keeps latency low,
- keeps behavior predictable and debuggable,
- provides a stable top match that the UI can auto-focus/highlight.

## What Is A Place-Recognition Model?

A place-recognition model is an image embedding model trained to match the same physical place across viewpoint/time/weather changes while separating different places.

Compared to generic CLIP-style retrieval:

- it is usually more sensitive to structural location cues (building geometry, facade layout, street furniture),
- and less reliant on broad semantic similarity (for example, “urban road” vs a specific corner).

In this pipeline, the place branch is used as a complementary signal, not a full replacement. This is why model weighting exists.

## Why The Architecture Is This Way

- **Multi-crop**: improves robustness for partial screenshots.
- **Multi-model**: reduces single-model failure modes.
- **Score fusion**: combines model evidence while keeping output simple.

This sequence is intentionally staged for fast query latency:

1. vector retrieval first (fast),
2. lightweight score fusion on returned candidates.

## Operational Notes

- Embeddings are generated during ingest and can be backfilled.
- Index stats report per-model coverage.
- If one model fails to load/encode, the pipeline can continue with remaining models (degraded mode).
- The UI is search-only and highlights the top match automatically.

## Current Limits

- Model fusion weights are heuristic (not yet learned).
- ORB/RANSAC can underperform on low-texture or motion-blurred crops.
- At very large scale, ANN index tuning and disk pressure heavily affect latency.

## Recommended Next Upgrades

1. learned fusion/calibration from labeled hard negatives,
2. stronger dedicated place-rec backbone,
3. stricter eval harness with partial-crop benchmark sets,
4. ANN/index tuning per model for large datasets.
