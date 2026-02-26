"""Configuration for the Street View crawler."""

import os
from dataclasses import dataclass, field
from typing import List
from urllib.parse import quote


@dataclass
class CrawlerConfig:
    # --- Deduplication ---
    # Minimum meters between captured panoramas
    DEDUP_RADIUS_METERS: float = 25.0

    # --- Capture settings ---
    # Headings (degrees) to capture at each panorama location
    # 0=North, 90=East, 180=South, 270=West
    HEADINGS: List[float] = field(default_factory=lambda: [0, 90, 180, 270])

    # Street View tilt angle in URL "...,{pitch}t".
    # Around ~75 points roughly toward the horizon in Google Maps Street View.
    PITCH: float = 75.0

    # Field of view (zoom level, lower = more zoomed in)
    FOV: float = 90.0

    # Seconds to wait for Street View to fully render before capturing
    CAPTURE_DELAY: float = 2.0

    # Viewport size for the browser
    VIEWPORT_WIDTH: int = 1920
    VIEWPORT_HEIGHT: int = 1080

    # --- Crawl limits ---
    MAX_CAPTURES: int = 10000  # Stop after this many panoramas
    MAX_RADIUS_KM: float = 5.0  # Don't wander further than this from seed

    # --- Navigation ---
    # 'bfs' = breadth-first (explore outward evenly)
    # 'dfs' = depth-first (follow streets deep)
    # 'random' = random walk
    NAV_STRATEGY: str = "bfs"

    # --- Paths ---
    DATABASE_URL: str = field(
        default_factory=lambda: os.getenv(
            "DATABASE_URL",
            "postgresql://geospy:geospy@127.0.0.1:5432/geospy",
        )
    )
    CAPTURES_DIR: str = "captures"

    # --- Rate limiting ---
    # Delay between navigation clicks (be nice to Google)
    NAV_DELAY: float = 1.5

    # --- Street View URL template ---
    # cbll/cbp form is the non-panoid entrypoint (opens Street View layer).
    SV_URL_TEMPLATE: str = (
        "https://www.google.com/maps?layer=c&cbll={lat},{lng}&cbp=12,{heading},0,0,0"
    )

    def get_streetview_url(
        self, lat: float, lng: float, heading: float = 0, pano_id: str = ""
    ) -> str:
        """
        Build a Street View URL.
        If pano_id is provided, use a direct panorama URL form:
        .../data=!3m7!1e1!3m5!1s{pano_id}!2e0!6s{thumb}!7i16384!8i8192
        """
        if pano_id:
            # Mirror the direct Street View share URL shape.
            thumb = (
                "https://streetviewpixels-pa.googleapis.com/v1/thumbnail"
                f"?cb_client=maps_sv.tactile&w=900&h=600"
                f"&pitch={self.PITCH}&panoid={pano_id}&yaw={heading}"
            )
            thumb_encoded = quote(thumb, safe="")
            return (
                "https://www.google.com/maps/"
                f"@{lat},{lng},3a,{self.FOV}y,{heading}h,{self.PITCH}t/"
                f"data=!3m7!1e1!3m5!1s{pano_id}!2e0!6s{thumb_encoded}!7i16384!8i8192"
            )

        return self.SV_URL_TEMPLATE.format(
            lat=lat,
            lng=lng,
            heading=heading,
        )
