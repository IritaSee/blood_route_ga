#!/usr/bin/env python3
"""Dump extracted locations and geocode status to CSV.

This utility helps diagnose GA feasibility failures caused by missing coordinates.
It can run in:
- cache-only mode (default): report what is already resolved in geocode cache
- resolve mode: attempt to geocode missing locations, then export final status
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

# Ensure project root is importable when running this script directly.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from optimizer.data_extractor import DataExtractor
from optimizer.geocoder import Geocoder


def _query_primary(name: str, district: str) -> str:
    # Matches geocoder.geocode_batch primary query construction.
    address = f"{name}, {district}" if district else name
    return f"{address}, Malang, ID"


def _query_fallback(name: str) -> str:
    # Matches geocoder.geocode_batch fallback query construction.
    return f"{name}, Malang, ID"


def _resolve_from_cache(geocoder: Geocoder, name: str, district: str) -> Tuple[Optional[float], Optional[float], str]:
    primary_q = _query_primary(name, district)
    fallback_q = _query_fallback(name)

    primary = geocoder.cache.get(primary_q)
    if primary and primary.get("success"):
        return primary.get("lat"), primary.get("lon"), "cache_primary"

    fallback = geocoder.cache.get(fallback_q)
    if fallback and fallback.get("success"):
        return fallback.get("lat"), fallback.get("lon"), "cache_fallback"

    if primary and primary.get("success") is False:
        return None, None, "cache_primary_failed"

    if fallback and fallback.get("success") is False:
        return None, None, "cache_fallback_failed"

    return None, None, "not_in_cache"


def _resolve_live(geocoder: Geocoder, name: str, district: str) -> Tuple[Optional[float], Optional[float], str]:
    address = f"{name}, {district}" if district else name
    result = geocoder.geocode(address)
    if result:
        return result.get("lat"), result.get("lon"), "live_primary"

    result = geocoder.geocode(name)
    if result:
        return result.get("lat"), result.get("lon"), "live_fallback"

    return None, None, "live_failed"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export location names with lat/lon and missing-coordinate status"
    )
    parser.add_argument("--pmi-file", default="data/Data PMI.xlsx")
    parser.add_argument("--droping-file", default="data/All Droping.xlsx")
    parser.add_argument(
        "--cache-file",
        default="results/historical_eval/geocode_cache.db",
        help="Path to geocode cache DB to inspect",
    )
    parser.add_argument(
        "--output",
        default="results/historical_eval/geocode_gap_report.csv",
        help="Output CSV path",
    )
    parser.add_argument(
        "--resolve-missing",
        action="store_true",
        help="Try live geocoding for locations not resolved from cache",
    )
    args = parser.parse_args()

    extractor = DataExtractor(pmi_file=args.pmi_file, droping_file=args.droping_file)
    locations = extractor.get_all_locations()

    geocoder = Geocoder(cache_file=args.cache_file)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    missing = 0
    for idx, loc in enumerate(locations):
        name = str(loc.get("name", "")).strip()
        district = str(loc.get("location_district", "")).strip()
        loc_type = str(loc.get("type", "")).strip()

        lat, lon, status = _resolve_from_cache(geocoder, name, district)
        if lat is None and lon is None and args.resolve_missing:
            lat, lon, status = _resolve_live(geocoder, name, district)

        has_coordinate = lat is not None and lon is not None
        if not has_coordinate:
            missing += 1

        rows.append(
            {
                "index": idx,
                "name": name,
                "location_district": district,
                "type": loc_type,
                "lat": lat,
                "lon": lon,
                "has_coordinate": has_coordinate,
                "status": status,
                "primary_query": _query_primary(name, district),
                "fallback_query": _query_fallback(name),
            }
        )

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "index",
                "name",
                "location_district",
                "type",
                "lat",
                "lon",
                "has_coordinate",
                "status",
                "primary_query",
                "fallback_query",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    total = len(rows)
    resolved = total - missing
    print(f"Wrote: {output_path}")
    print(f"Locations total: {total}")
    print(f"Resolved coordinates: {resolved}")
    print(f"Missing coordinates: {missing}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
