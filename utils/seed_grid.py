"""
Utility to generate a grid of seed coordinates over a bounding box.
Useful for systematic coverage of a city/area instead of relying
solely on Street View navigation links.

Usage:
    python seed_grid.py --bbox 37.70,-122.52,37.82,-122.35 --step 100
    # Outputs seeds.csv with lat,lon pairs 100m apart covering SF
"""

import argparse
import csv
import math


def generate_grid(
    min_lat: float,
    min_lon: float,
    max_lat: float,
    max_lon: float,
    step_meters: float = 100
) -> list:
    """Generate a grid of (lat, lon) points within a bounding box."""
    points = []

    # Convert step to degrees
    lat_step = step_meters / 111320.0
    # Use center latitude for lon step
    center_lat = (min_lat + max_lat) / 2
    lon_step = step_meters / (111320.0 * math.cos(math.radians(center_lat)))

    lat = min_lat
    while lat <= max_lat:
        lon = min_lon
        while lon <= max_lon:
            points.append((round(lat, 6), round(lon, 6)))
            lon += lon_step
        lat += lat_step

    return points


def main():
    parser = argparse.ArgumentParser(description="Generate seed grid for crawler")
    parser.add_argument(
        "--bbox",
        type=str,
        required=True,
        help="Bounding box: min_lat,min_lon,max_lat,max_lon",
    )
    parser.add_argument(
        "--step",
        type=float,
        default=100,
        help="Grid step in meters (default: 100)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="seeds.csv",
        help="Output CSV file",
    )

    args = parser.parse_args()
    try:
        bbox = [float(x) for x in args.bbox.split(",")]
    except ValueError:
        parser.error("Bounding box must be numeric: min_lat,min_lon,max_lat,max_lon")
    if len(bbox) != 4:
        parser.error("Bounding box must have 4 values: min_lat,min_lon,max_lat,max_lon")
    if args.step <= 0:
        parser.error("--step must be > 0")
    if bbox[0] > bbox[2] or bbox[1] > bbox[3]:
        parser.error("Bounding box must be ordered as min_lat,min_lon,max_lat,max_lon")

    points = generate_grid(bbox[0], bbox[1], bbox[2], bbox[3], args.step)

    with open(args.output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["lat", "lon"])
        writer.writerows(points)

    print(f"Generated {len(points)} seed points -> {args.output}")
    area_km2 = (
        (bbox[2] - bbox[0]) * 111.32 * (bbox[3] - bbox[1]) * 111.32 * math.cos(math.radians((bbox[0] + bbox[2]) / 2))
    )
    print(f"Area: ~{area_km2:.1f} km^2")
    density = (len(points) / area_km2) if area_km2 > 0 else 0
    print(f"Grid density: {density:.0f} points/km^2 at {args.step}m spacing")


if __name__ == "__main__":
    main()

