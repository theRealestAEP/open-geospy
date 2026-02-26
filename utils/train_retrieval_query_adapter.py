"""Train a query-side linear adapter from hard-negative triplets."""

import argparse
import csv
import io
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from PIL import Image

from backend.app.clip_embeddings import ClipEmbedder, get_retrieval_embedders
from config import CrawlerConfig
from db.postgres_database import Database


@dataclass
class TripletRow:
    query_capture_id: int
    query_variant: str
    positive_capture_id: int
    negative_capture_id: int
    mode: str


def _capture_abs_path(captures_dir: str, filepath: str) -> str:
    raw = (filepath or "").strip()
    if not raw:
        return ""
    if os.path.isabs(raw):
        return raw
    normalized = raw.replace("\\", "/")
    if normalized.startswith("captures/"):
        return os.path.join(captures_dir, normalized[len("captures/") :])
    return os.path.abspath(raw)


def _jpeg_bytes(image: Image.Image) -> bytes:
    out = io.BytesIO()
    image.convert("RGB").save(out, format="JPEG", quality=92)
    return out.getvalue()


def _crop_variant(image_bytes: bytes, variant: str) -> Optional[bytes]:
    key = str(variant or "full").strip().lower()
    with Image.open(io.BytesIO(image_bytes)) as img:
        img = img.convert("RGB")
        w, h = img.size

        def crop(box: Tuple[int, int, int, int]) -> Optional[bytes]:
            x0, y0, x1, y1 = box
            if x1 - x0 < 24 or y1 - y0 < 24:
                return None
            return _jpeg_bytes(img.crop(box))

        if key == "full":
            return image_bytes
        if key == "center80":
            cw, ch = int(w * 0.8), int(h * 0.8)
            return crop(((w - cw) // 2, (h - ch) // 2, (w + cw) // 2, (h + ch) // 2))
        if key == "center60":
            cw, ch = int(w * 0.6), int(h * 0.6)
            return crop(((w - cw) // 2, (h - ch) // 2, (w + cw) // 2, (h + ch) // 2))
        if key == "center40":
            cw, ch = int(w * 0.4), int(h * 0.4)
            return crop(((w - cw) // 2, (h - ch) // 2, (w + cw) // 2, (h + ch) // 2))
        if key == "left":
            return crop((0, 0, w // 2, h))
        if key == "right":
            return crop((w // 2, 0, w, h))
        if key == "top":
            return crop((0, 0, w, h // 2))
        if key == "bottom":
            return crop((0, h // 2, w, h))
        if key == "upper_center":
            return crop((w // 4, 0, (3 * w) // 4, (2 * h) // 3))
        if key == "lower_center":
            return crop((w // 4, h // 3, (3 * w) // 4, h))
        if key == "q1":
            return crop((0, 0, w // 2, h // 2))
        if key == "q2":
            return crop((w // 2, 0, w, h // 2))
        if key == "q3":
            return crop((0, h // 2, w // 2, h))
        if key == "q4":
            return crop((w // 2, h // 2, w, h))
    return None


def _read_hard_negatives(
    path: str,
    *,
    mode_filter: str,
    max_triplets: int,
    seed: int,
) -> List[TripletRow]:
    rows: List[TripletRow] = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                mode = str(row.get("mode", "")).strip().lower()
                if mode_filter != "both" and mode != mode_filter:
                    continue
                query_capture_id = int(row.get("query_capture_id", "0"))
                positive_capture_id = int(row.get("expected_capture_id", "0"))
                negative_capture_id = int(row.get("pred_capture_id", "0"))
                query_variant = str(row.get("query_variant", "full")).strip().lower()
                if query_capture_id <= 0 or positive_capture_id <= 0 or negative_capture_id <= 0:
                    continue
                if positive_capture_id == negative_capture_id:
                    continue
                rows.append(
                    TripletRow(
                        query_capture_id=query_capture_id,
                        query_variant=query_variant,
                        positive_capture_id=positive_capture_id,
                        negative_capture_id=negative_capture_id,
                        mode=mode or "unknown",
                    )
                )
            except Exception:
                continue
    random.Random(seed).shuffle(rows)
    if max_triplets > 0:
        rows = rows[:max_triplets]
    return rows


def _chunked(seq: Sequence[int], size: int) -> List[List[int]]:
    out: List[List[int]] = []
    for i in range(0, len(seq), size):
        out.append(list(seq[i : i + size]))
    return out


def _fetch_capture_paths(db: Database, capture_ids: Sequence[int]) -> Dict[int, str]:
    if not capture_ids:
        return {}
    rows: Dict[int, str] = {}
    for chunk in _chunked(list(dict.fromkeys(int(x) for x in capture_ids)), 4000):
        sql = (
            "SELECT id, filepath FROM captures "
            "WHERE id = ANY(%s)"
        )
        for row in db.conn.execute(sql, (chunk,)).fetchall():
            rows[int(row["id"])] = str(row.get("filepath", "") or "")
    return rows


def _parse_vector_text(value: str) -> List[float]:
    raw = str(value or "").strip()
    if raw.startswith("[") and raw.endswith("]"):
        raw = raw[1:-1]
    if not raw:
        return []
    return [float(x) for x in raw.split(",") if x.strip()]


def _fetch_embeddings(
    db: Database,
    capture_ids: Sequence[int],
    *,
    model_name: str,
    model_version: str,
) -> Dict[int, List[float]]:
    if not capture_ids:
        return {}
    out: Dict[int, List[float]] = {}
    for chunk in _chunked(list(dict.fromkeys(int(x) for x in capture_ids)), 2000):
        rows = db.conn.execute(
            """
            SELECT capture_id, embedding::text AS embedding_text
            FROM capture_embeddings
            WHERE model_name = %s
              AND model_version = %s
              AND capture_id = ANY(%s)
            """,
            (model_name, model_version, chunk),
        ).fetchall()
        for row in rows:
            vec = _parse_vector_text(str(row.get("embedding_text", "")))
            if vec:
                out[int(row["capture_id"])] = vec
    return out


def _pick_device(torch_module):
    if torch_module.cuda.is_available():
        return "cuda"
    if hasattr(torch_module.backends, "mps") and torch_module.backends.mps.is_available():
        return "mps"
    return "cpu"


def _batch_encode_queries(
    embedder: ClipEmbedder,
    query_payloads: List[Tuple[int, str, bytes]],
    *,
    batch_size: int,
) -> Dict[Tuple[int, str], List[float]]:
    out: Dict[Tuple[int, str], List[float]] = {}
    for i in range(0, len(query_payloads), batch_size):
        batch = query_payloads[i : i + batch_size]
        vectors = embedder.encode_image_bytes_batch([item[2] for item in batch])
        for (capture_id, variant, _), vec in zip(batch, vectors):
            out[(capture_id, variant)] = vec
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Train a query-side adapter using hard negatives from partial eval."
    )
    parser.add_argument("--hard-negatives-csv", required=True)
    parser.add_argument("--output", default="artifacts/query_adapter_clip.pt")
    parser.add_argument("--model-id", default="clip")
    parser.add_argument("--mode", choices=["search", "locate", "both"], default="both")
    parser.add_argument("--max-triplets", type=int, default=80000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--margin", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--identity-regularization", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(int(args.seed))

    os.environ.pop("GEOSPY_RETRIEVAL_QUERY_ADAPTER_PATH", None)
    for key in list(os.environ.keys()):
        if key.startswith("GEOSPY_RETRIEVAL_QUERY_ADAPTER_") and key.endswith("_PATH"):
            os.environ.pop(key, None)

    embedders = list(get_retrieval_embedders())
    selected = None
    for embedder in embedders:
        if str(getattr(embedder, "model_id", "")).lower() == str(args.model_id).lower():
            selected = embedder
            break
    if selected is None:
        raise SystemExit(f"Model id not found: {args.model_id}")

    cfg = CrawlerConfig()
    captures_dir = cfg.CAPTURES_DIR
    if not os.path.isabs(captures_dir):
        captures_dir = os.path.abspath(captures_dir)

    triplets = _read_hard_negatives(
        args.hard_negatives_csv,
        mode_filter=args.mode,
        max_triplets=max(1, int(args.max_triplets)),
        seed=int(args.seed),
    )
    if not triplets:
        raise SystemExit("No valid hard negatives found.")
    print(f"loaded_triplets={len(triplets)}")

    db = Database(cfg.DATABASE_URL)
    try:
        query_capture_ids = [row.query_capture_id for row in triplets]
        query_paths = _fetch_capture_paths(db, query_capture_ids)
        query_payloads: List[Tuple[int, str, bytes]] = []
        seen_query_keys = set()
        for row in triplets:
            key = (row.query_capture_id, row.query_variant)
            if key in seen_query_keys:
                continue
            seen_query_keys.add(key)
            raw_path = query_paths.get(row.query_capture_id, "")
            abs_path = _capture_abs_path(captures_dir, raw_path)
            if not abs_path or not os.path.exists(abs_path):
                continue
            try:
                with open(abs_path, "rb") as f:
                    source = f.read()
                cropped = _crop_variant(source, row.query_variant)
                if not cropped:
                    continue
                query_payloads.append((row.query_capture_id, row.query_variant, cropped))
            except Exception:
                continue
        if not query_payloads:
            raise SystemExit("No query crops could be built from hard negatives.")
        print(f"query_crops={len(query_payloads)}")

        query_vectors = _batch_encode_queries(
            selected, query_payloads, batch_size=max(8, int(args.batch_size))
        )
        if not query_vectors:
            raise SystemExit("Failed to encode query crops.")

        pos_neg_ids = [row.positive_capture_id for row in triplets] + [
            row.negative_capture_id for row in triplets
        ]
        embedding_map = _fetch_embeddings(
            db,
            pos_neg_ids,
            model_name=selected.model_name,
            model_version=selected.model_version,
        )
        if not embedding_map:
            raise SystemExit("Failed to load candidate embeddings for hard negatives.")

        q_vectors: List[List[float]] = []
        p_vectors: List[List[float]] = []
        n_vectors: List[List[float]] = []
        for row in triplets:
            qv = query_vectors.get((row.query_capture_id, row.query_variant))
            pv = embedding_map.get(row.positive_capture_id)
            nv = embedding_map.get(row.negative_capture_id)
            if not qv or not pv or not nv:
                continue
            if len(qv) != len(pv) or len(qv) != len(nv):
                continue
            q_vectors.append(qv)
            p_vectors.append(pv)
            n_vectors.append(nv)
    finally:
        db.close()

    if not q_vectors:
        raise SystemExit("No valid training triplets after filtering.")
    print(f"train_triplets={len(q_vectors)}")

    import torch
    import torch.nn.functional as F

    torch.manual_seed(int(args.seed))
    device = _pick_device(torch)
    print(f"device={device}")

    q = torch.tensor(q_vectors, dtype=torch.float32, device=device)
    p = torch.tensor(p_vectors, dtype=torch.float32, device=device)
    n = torch.tensor(n_vectors, dtype=torch.float32, device=device)
    q = F.normalize(q, dim=-1)
    p = F.normalize(p, dim=-1)
    n = F.normalize(n, dim=-1)
    dim = int(q.shape[1])

    adapter = torch.nn.Linear(dim, dim, bias=False, device=device)
    with torch.no_grad():
        adapter.weight.copy_(torch.eye(dim, device=device))
    optimizer = torch.optim.AdamW(
        adapter.parameters(),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
    )
    margin = float(args.margin)
    id_lambda = max(0.0, float(args.identity_regularization))
    batch_size = max(8, int(args.batch_size))

    for epoch in range(1, max(1, int(args.epochs)) + 1):
        order = torch.randperm(q.shape[0], device=device)
        epoch_loss = 0.0
        epoch_margin_ok = 0.0
        count = 0
        for start in range(0, q.shape[0], batch_size):
            idx = order[start : start + batch_size]
            qb = q[idx]
            pb = p[idx]
            nb = n[idx]
            q_proj = F.normalize(adapter(qb), dim=-1)
            sim_pos = (q_proj * pb).sum(dim=-1)
            sim_neg = (q_proj * nb).sum(dim=-1)
            margin_loss = torch.relu(margin - sim_pos + sim_neg).mean()
            identity_loss = ((adapter.weight - torch.eye(dim, device=device)) ** 2).mean()
            loss = margin_loss + (id_lambda * identity_loss)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                epoch_loss += float(loss.item()) * len(idx)
                epoch_margin_ok += float((sim_pos > sim_neg).float().sum().item())
                count += len(idx)
        avg_loss = epoch_loss / max(1, count)
        margin_acc = epoch_margin_ok / max(1, count)
        print(
            f"epoch={epoch} loss={avg_loss:.6f} margin_acc={margin_acc:.4f} samples={count}"
        )

    out_path = os.path.abspath(str(args.output))
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    checkpoint = {
        "state_dict": adapter.cpu().state_dict(),
        "dim": dim,
        "model_id": selected.model_id,
        "model_name": selected.model_name,
        "model_version": selected.model_version,
        "train_triplets": len(q_vectors),
        "epochs": int(args.epochs),
        "margin": margin,
        "lr": float(args.lr),
        "identity_regularization": id_lambda,
    }
    torch.save(checkpoint, out_path)
    print(f"saved_adapter={out_path}")
    print(
        f"enable_with=GEOSPY_RETRIEVAL_QUERY_ADAPTER_{selected.model_id.upper()}_PATH={out_path}"
    )


if __name__ == "__main__":
    main()

