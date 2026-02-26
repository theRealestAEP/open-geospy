"""
Filter seed CSV points by proximity to OSM road geometry.

This is a practical ocean/water prefilter for batch crawling:
points far from roads are removed before workers start.

Usage:
    python seed_filter_roads.py --input sf_city_seeds.csv --output sf_land_seeds.csv --near-road 90
"""

import argparse
import csv
import json
import math
from collections import defaultdict
from typing import Dict, Iterable, List, Tuple
from urllib.parse import quote
from urllib.request import urlopen


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371000.0
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def load_seeds(path: str) -> List[Tuple[float, float]]:
    with open(path, "r", newline="") as f:
        r = csv.DictReader(f)
        if "lat" not in (r.fieldnames or []) or "lon" not in (r.fieldnames or []):
            raise ValueError("CSV must have lat,lon headers")
        return [(float(row["lat"]), float(row["lon"])) for row in r]


def bounds(points: Iterable[Tuple[float, float]]) -> Tuple[float, float, float, float]:
    lats = [p[0] for p in points]
    lons = [p[1] for p in points]
    return min(lats), min(lons), max(lats), max(lons)


def fetch_osm_road_nodes(
    min_lat: float, min_lon: float, max_lat: float, max_lon: float, timeout_s: int = 180
) -> List[Tuple[float, float]]:
    query = f"""
[out:json][timeout:120];
(
  way["highway"]({min_lat},{min_lon},{max_lat},{max_lon});
);
out geom;
"""
    url = "https://overpass-api.de/api/interpreter?data=" + quote(query.strip())
    with urlopen(url, timeout=timeout_s) as resp:
        payload = json.loads(resp.read().decode("utf-8"))

    nodes: List[Tuple[float, float]] = []
    for el in payload.get("elements", []):
        if el.get("type") != "way":
            continue
        for g in el.get("geometry", []):
            lat = g.get("lat")
            lon = g.get("lon")
            if lat is not None and lon is not None:
                nodes.append((float(lat), float(lon)))
    return nodes


def bucket_key(lat: float, lon: float, cell_deg: float) -> Tuple[int, int]:
    return (int(math.floor(lat / cell_deg)), int(math.floor(lon / cell_deg)))


def build_spatial_index(
    nodes: List[Tuple[float, float]], cell_deg: float
) -> Dict[Tuple[int, int], List[Tuple[float, float]]]:
    idx: Dict[Tuple[int, int], List[Tuple[float, float]]] = defaultdict(list)
    for lat, lon in nodes:
        idx[bucket_key(lat, lon, cell_deg)].append((lat, lon))
    return idx


def near_any_road(
    lat: float,
    lon: float,
    idx: Dict[Tuple[int, int], List[Tuple[float, float]]],
    cell_deg: float,
    max_dist_m: float,
) -> bool:
    kx, ky = bucket_key(lat, lon, cell_deg)
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            pts = idx.get((kx + dx, ky + dy))
            if not pts:
                continue
            for rlat, rlon in pts:
                if haversine_m(lat, lon, rlat, rlon) <= max_dist_m:
                    return True
    return False


def main():
    p = argparse.ArgumentParser(description="Filter seeds to points near roads")
    p.add_argument("--input", required=True, help="Input CSV with lat,lon")
    p.add_argument("--output", required=True, help="Output filtered CSV")
    p.add_argument(
        "--near-road",
        type=float,
        default=90.0,
        help="Keep points within this many meters of any road node (default: 90)",
    )
    p.add_argument(
        "--bbox-pad-m",
        type=float,
        default=300.0,
        help="Padding around seed bbox when querying OSM roads (default: 300m)",
    )
    args = p.parse_args()

    if args.near_road <= 0:
        raise SystemExit("--near-road must be > 0")

    seeds = load_seeds(args.input)
    if not seeds:
        raise SystemExit("No seeds found in input")

    min_lat, min_lon, max_lat, max_lon = bounds(seeds)
    pad_lat = args.bbox_pad_m / 111320.0
    center_lat = (min_lat + max_lat) / 2.0
    pad_lon = args.bbox_pad_m / (111320.0 * max(0.01, abs(math.cos(math.radians(center_lat)))))

    q_min_lat = min_lat - pad_lat
    q_min_lon = min_lon - pad_lon
    q_max_lat = max_lat + pad_lat
    q_max_lon = max_lon + pad_lon

    print(
        "Fetching OSM roads for bbox:",
        f"{q_min_lat:.6f},{q_min_lon:.6f},{q_max_lat:.6f},{q_max_lon:.6f}",
    )
    road_nodes = fetch_osm_road_nodes(q_min_lat, q_min_lon, q_max_lat, q_max_lon)
    if not road_nodes:
        raise SystemExit("No road nodes returned from OSM; aborting")

    # ~45m cells by default for quick candidate lookup.
    cell_deg = min(0.001, max(0.0002, (args.near_road / 2.0) / 111320.0))
    idx = build_spatial_index(road_nodes, cell_deg)

    kept: List[Tuple[float, float]] = []
    for lat, lon in seeds:
        if near_any_road(lat, lon, idx, cell_deg, args.near_road):
            kept.append((lat, lon))

    with open(args.output, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["lat", "lon"])
        w.writerows((round(a, 6), round(b, 6)) for a, b in kept)

    dropped = len(seeds) - len(kept)
    print(
        f"Input: {len(seeds)} | Kept: {len(kept)} | Dropped: {dropped} "
        f"({(dropped / len(seeds)) * 100:.1f}%)"
    )
    print(f"Saved filtered seeds to {args.output}")


if __name__ == "__main__":
    main()

