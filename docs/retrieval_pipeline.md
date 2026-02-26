# Retrieval Pipeline Write-Up

This document explains how GeoSpy retrieval works today, why each stage exists, and where the place-recognition model fits in.

## What The Pipeline Does

Given a query image, the system tries to:

1. find similar captures in the DB,
2. combine evidence across crops and models,
3. reject visually/geometrically inconsistent matches,
4. produce a final location estimate with confidence and radius.

The same embedding stack is used for:

- `search-by-image` (ranked matches),
- `locate-by-image` (location estimate),
- background/index backfill pipelines.

## High-Level Flow

```mermaid
flowchart LR
queryImage[QueryImageUpload] --> cropBuilder[MultiCropBuilder]
cropBuilder --> clipModel[ClipEmbedder]
cropBuilder --> placeModel[PlaceRecEmbedder]
clipModel --> clipSearch[PgVectorSearchClip]
placeModel --> placeSearch[PgVectorSearchPlace]
clipSearch --> candidateFusion[CandidateFusionAndVoteCap]
placeSearch --> candidateFusion
candidateFusion --> modalRerank[OptionalModalClipRerank]
modalRerank --> geoCluster[GeoClustering]
geoCluster --> geomVerify[GeometricVerificationOrbOrLightGlue]
geomVerify --> appearanceCheck[AppearancePenalty]
appearanceCheck --> finalScore[HybridScoringAndConfidence]
finalScore --> estimateOutput[BestEstimateAndSupportingMatches]
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

### 3) Candidate fusion + panorama vote cap

All candidate rows are merged by capture id, weighted by model and crop hits. Then per-panorama contributions are capped.

**Why**:
- fusion stabilizes ranking across noisy crops,
- vote cap prevents one panorama from dominating due to many near-duplicate headings.

### 4) Geo clustering

Candidates are grouped by geographic proximity and rescored by cluster evidence.

**Why**:
- true matches tend to form local geographic consensus,
- random false positives are often spatially scattered.

### 5) Geometric verification (ORB + RANSAC, optional Modal LightGlue)

Top candidates are image-verified with local keypoints and homography inliers.

**Why**:
- embedding similarity alone can match wrong but semantically similar scenes,
- geometric consistency tests whether structures align in image space,
- LightGlue verification in Modal improves hard partials where ORB is weak.

### 6) Appearance penalty

A lightweight appearance mismatch penalty is applied for obvious color/composition mismatch.

**Why**:
- helps reject clearly different facades when semantic embedding still scores high.

### 7) Final scoring + confidence/radius

Final score combines vector evidence, geometric score, and penalties. Output includes:

- best estimate (`lat/lon`),
- confidence (`0..1`),
- uncertainty radius,
- supporting matches and debug diagnostics.

**Why**:
- location decisions should include uncertainty, not just a single point.

## What Is A Place-Recognition Model?

A place-recognition model is an image embedding model trained to match the same physical place across viewpoint/time/weather changes while separating different places.

Compared to generic CLIP-style retrieval:

- it is usually more sensitive to structural location cues (building geometry, facade layout, street furniture),
- and less reliant on broad semantic similarity (for example, “urban road” vs a specific corner).

In this pipeline, the place branch is used as a complementary signal, not a full replacement. This is why model weighting exists.

## Why The Architecture Is This Way

- **Multi-crop**: improves robustness for partial screenshots.
- **Multi-model**: reduces single-model failure modes.
- **Vote cap + clustering**: enforces spatial consensus.
- **Geometric check**: removes visually inconsistent false positives.
- **Confidence/radius output**: makes downstream decisions safer than top-1 only.

This sequence is intentionally staged from cheap to expensive:

1. vector retrieval first (fast),
2. rerank/geom verification only on narrowed candidates (costly),
3. final confidence on already filtered evidence.

## Operational Notes

- Embeddings are generated during ingest and can be backfilled.
- Index stats report per-model coverage.
- If one model fails to load/encode, the pipeline can continue with remaining models (degraded mode).
- Locate tuning is request-level: UI fetches defaults/bounds from `GET /api/retrieval/locate-params` and sends knobs with each `POST /api/retrieval/locate-by-image`.

## Current Limits

- Model fusion weights are heuristic (not yet learned).
- ORB/RANSAC can underperform on low-texture or motion-blurred crops.
- At very large scale, ANN index tuning and disk pressure heavily affect latency.

## Recommended Next Upgrades

1. learned fusion/calibration from labeled hard negatives,
2. stronger dedicated place-rec backbone,
3. stricter eval harness with partial-crop benchmark sets,
4. ANN/index tuning per model for large datasets.
