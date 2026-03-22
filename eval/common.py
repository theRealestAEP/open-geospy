import csv
import json
import math
import os
from dataclasses import asdict, dataclass
from typing import Iterable, List, Optional


@dataclass
class EvalCase:
    case_id: str
    image_path: str
    expected_lat: Optional[float] = None
    expected_lon: Optional[float] = None
    expected_panorama_id: Optional[int] = None
    expected_capture_id: Optional[int] = None
    expected_reject: bool = False
    split: str = "dev"
    notes: str = ""


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius_m = 6371000.0
    phi1 = math.radians(float(lat1))
    phi2 = math.radians(float(lat2))
    dphi = math.radians(float(lat2) - float(lat1))
    dlambda = math.radians(float(lon2) - float(lon1))
    a = (
        math.sin(dphi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * (math.sin(dlambda / 2.0) ** 2)
    )
    return 2.0 * radius_m * math.atan2(math.sqrt(a), math.sqrt(max(1e-12, 1.0 - a)))


def _parse_optional_float(value: str) -> Optional[float]:
    raw = str(value or "").strip()
    if not raw:
        return None
    return float(raw)


def _parse_optional_int(value: str) -> Optional[int]:
    raw = str(value or "").strip()
    if not raw:
        return None
    return int(raw)


def _parse_bool(value: str) -> bool:
    raw = str(value or "").strip().lower()
    return raw in {"1", "true", "yes", "y", "on"}


def load_eval_cases(csv_path: str) -> List[EvalCase]:
    items: List[EvalCase] = []
    base_dir = os.path.abspath(os.path.dirname(csv_path))
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader, start=1):
            case_id = str(row.get("case_id") or f"case-{idx}").strip()
            raw_image_path = str(row.get("image_path") or "").strip()
            if not raw_image_path:
                continue
            image_path = (
                raw_image_path
                if os.path.isabs(raw_image_path)
                else os.path.abspath(os.path.join(base_dir, raw_image_path))
            )
            items.append(
                EvalCase(
                    case_id=case_id,
                    image_path=image_path,
                    expected_lat=_parse_optional_float(str(row.get("expected_lat", ""))),
                    expected_lon=_parse_optional_float(str(row.get("expected_lon", ""))),
                    expected_panorama_id=_parse_optional_int(
                        str(row.get("expected_panorama_id", ""))
                    ),
                    expected_capture_id=_parse_optional_int(
                        str(row.get("expected_capture_id", ""))
                    ),
                    expected_reject=_parse_bool(str(row.get("expected_reject", ""))),
                    split=str(row.get("split") or "dev").strip() or "dev",
                    notes=str(row.get("notes") or "").strip(),
                )
            )
    return items


def write_csv(path: str, rows: Iterable[dict]) -> None:
    rows = list(rows)
    if not rows:
        return
    keys = sorted({key for row in rows for key in row.keys()})
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: str, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def cases_to_rows(cases: Iterable[EvalCase]) -> List[dict]:
    return [asdict(item) for item in cases]
