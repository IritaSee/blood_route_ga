"""
Render an OSM map (Folium/Leaflet) comparing three route layers:

  - Historical baseline: depot <-> each historical destination (All Droping.xlsx
    trips are single-destination, not multi-stop tours).
  - OSRM-GA ("regular GA"): multi-stop tour per vehicle from run_historical.py's
    ga_results.json.
  - Google-GA ("GA + Maps API"): multi-stop tour per vehicle from run_live.py's
    ga_results.json.

Road-following geometry for drawing is fetched (and cached) via
optimizer/route_geometry.py - OSRM's /route service for the baseline and OSRM-GA
layers, Google's Directions API for the Google-GA layer, matching what actually
informed each route's cost.

Requires ga_results.json files that include a `stops` key (added to
optimizer/pipeline.py alongside `--seed` support) - re-run run_historical.py /
run_live.py if the existing results predate that change.

Usage:
    python scripts/visualize_routes.py
    python scripts/visualize_routes.py --historical-dir results/historical_eval_v3 \
        --live-dir results/live_eval_v3 --output results/route_comparison_map.html
"""

import argparse
import csv
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import folium

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from optimizer.baseline_extractor import BaselineExtractor
from optimizer.route_geometry import RouteGeometryFetcher

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

VEHICLE_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd', '#8c564b']
BASELINE_COLOR = '#999999'

# All Droping.xlsx's "Tujuan 1" column spells out facility names in full (e.g.
# "Bank Darah Rumah Sakit (BDRS) Kanjuruhan"), while data/geocode_overrides.csv
# uses the abbreviated names the rest of the pipeline geocodes against (e.g.
# "BDRS Kanjuruhan"). The abbreviation used isn't even consistent between sources
# (BDRS <-> UTD RS, BDRS <-> BRRS), so this can't be solved by normalization alone
# - it's a fixed, known set of aliases for the 9 facilities where the two sheets
# disagree.
BASELINE_NAME_ALIASES = {
    'Bank Darah Rumah Sakit (BDRS) Kanjuruhan': 'BDRS Kanjuruhan',
    'Bank Darah Rumah Sakit (BDRS) UMM': 'BDRS UMM',
    'Bank Darah Rumah Sakit (BDRS) Saiful Anwar': 'UTD RS Saiful Anwar',
    'Bank Darah Rumah Sakit (BDRS) Wava Husada': 'BRRS Wava Husada',
    'Bank Darah Rumah Sakit (BDRS) Karsa Husada': 'BDRS Karsa Husada',
    'UTD (Unit Transfusi Darah) Kota Kediri': 'UDD PMI Kota Kediri',
    'UTD (Unit Transfusi Darah) Kota Batu': 'UDD PMI Kota Batu',
    'UTD (Unit Transfusi Darah) Kab Wonogiri': 'UDD PMI Kabupaten Wonogiri',
    'UTD (Unit Transfusi Darah) Madiun': 'UDD PMI Kota Madiun',
}


def _normalize(name: str) -> str:
    return re.sub(r"\s+", " ", str(name or "").strip().lower())


def load_geocode_overrides(csv_path: Path) -> Dict[str, Tuple[float, float]]:
    overrides = {}
    with csv_path.open('r', encoding='utf-8', newline='') as fh:
        for row in csv.DictReader(fh):
            name = (row.get('name') or '').strip()
            lat, lon = row.get('lat'), row.get('lon')
            if not name or not lat or not lon:
                continue
            overrides[_normalize(name)] = (float(lat), float(lon))
    return overrides


def load_ga_stops(results_dir: Path) -> Tuple[List[List[Dict]], Dict]:
    """Load per-vehicle stop lists + summary metrics from a pipeline output dir."""
    ga_results_path = results_dir / "ga_results.json"
    if not ga_results_path.exists():
        raise FileNotFoundError(
            f"{ga_results_path} not found. Run the pipeline first, e.g.:\n"
            f"  python run_historical.py --seed 42 --output-dir {results_dir}"
        )
    data = json.loads(ga_results_path.read_text())
    if 'stops' not in data:
        raise ValueError(
            f"{ga_results_path} has no 'stops' key - it was generated before route "
            f"persistence was added to optimizer/pipeline.py. Re-run the pipeline "
            f"to regenerate it."
        )
    return data['stops'], data


def build_baseline_pairs(overrides: Dict[str, Tuple[float, float]], droping_file: str) -> List[Dict]:
    """One depot<->destination pair per historical destination, with trip stats."""
    extractor = BaselineExtractor(droping_file)
    dest_stats = extractor.get_destination_baseline()

    pairs = []
    unmatched = []
    for name, stats in dest_stats.items():
        lookup_name = BASELINE_NAME_ALIASES.get(name, name)
        coords = overrides.get(_normalize(lookup_name))
        if coords is None:
            unmatched.append(name)
            continue
        pairs.append({'name': name, 'lat': coords[0], 'lon': coords[1], 'stats': stats})

    if unmatched:
        logger.warning(f"{len(unmatched)} historical destinations have no geocode override "
                        f"match and were skipped: {unmatched}")
    logger.info(f"Baseline: {len(pairs)}/{len(dest_stats)} destinations matched to coordinates")
    return pairs


def add_baseline_layer(m: "folium.Map", depot: Tuple[float, float], pairs: List[Dict],
                        fetcher: RouteGeometryFetcher):
    layer = folium.FeatureGroup(name="Historical baseline (depot <-> destination)", show=True)
    for pair in pairs:
        geometry = fetcher.fetch_osrm([depot, (pair['lat'], pair['lon'])])
        stats = pair['stats']
        popup = (f"<b>{pair['name']}</b><br>"
                 f"{stats['num_trips']} historical trips<br>"
                 f"avg {stats['avg_distance_km']:.1f} km, "
                 f"{stats['on_time_percentage']:.0f}% on-time")
        folium.PolyLine(geometry, color=BASELINE_COLOR, weight=2, opacity=0.6,
                         dash_array="4,6", tooltip=pair['name'], popup=popup).add_to(layer)
        folium.CircleMarker((pair['lat'], pair['lon']), radius=3, color=BASELINE_COLOR,
                             fill=True, fill_opacity=0.8, popup=popup).add_to(layer)
    layer.add_to(m)


def add_ga_layer(m: "folium.Map", layer_name: str, stops_per_vehicle: List[List[Dict]],
                  summary: Dict, fetch_geometry: Callable[[List[Tuple[float, float]]], List[Tuple[float, float]]]):
    layer = folium.FeatureGroup(name=layer_name, show=True)
    for vehicle_id, stops in enumerate(stops_per_vehicle):
        if len(stops) <= 2:
            continue  # depot-only, no customers assigned to this vehicle
        coords = [(s['lat'], s['lon']) for s in stops if s['lat'] is not None and s['lon'] is not None]
        if len(coords) < 2:
            continue

        color = VEHICLE_COLORS[vehicle_id % len(VEHICLE_COLORS)]
        geometry = fetch_geometry(coords)

        distance_km = summary['vehicle_distances_km'][vehicle_id]
        time_h = summary['vehicle_times_hours'][vehicle_id]
        cost_idr = summary['vehicle_costs_idr'][vehicle_id]
        popup = (f"<b>{layer_name} - Vehicle {vehicle_id + 1}</b><br>"
                 f"{distance_km:.1f} km, {time_h:.1f} h<br>"
                 f"IDR {cost_idr:,.0f}<br>"
                 f"{len(stops) - 2} stops")

        folium.PolyLine(geometry, color=color, weight=4, opacity=0.85,
                         tooltip=f"{layer_name} - Vehicle {vehicle_id + 1}", popup=popup).add_to(layer)

        for i, s in enumerate(stops):
            if s['lat'] is None or s['lon'] is None:
                continue
            is_depot = i == 0 or i == len(stops) - 1
            folium.CircleMarker(
                (s['lat'], s['lon']),
                radius=6 if is_depot else 4,
                color=color,
                fill=True,
                fill_color='black' if is_depot else color,
                fill_opacity=1.0,
                popup=s['name'],
            ).add_to(layer)

    layer.add_to(m)


def main():
    parser = argparse.ArgumentParser(description="Visualize and compare GA routes on an OSM map")
    parser.add_argument("--historical-dir", default="results/historical_eval_v3",
                         help="Pipeline output dir from run_historical.py (OSRM-GA)")
    parser.add_argument("--live-dir", default="results/live_eval_v3",
                         help="Pipeline output dir from run_live.py (Google-GA)")
    parser.add_argument("--droping-file", default="data/All Droping.xlsx")
    parser.add_argument("--overrides-file", default="data/geocode_overrides.csv")
    parser.add_argument("--output", default="results/route_comparison_map.html")
    parser.add_argument("--geometry-cache", default="results/route_geometry_cache.db")
    parser.add_argument("--skip-baseline", action="store_true", help="Skip the historical baseline layer")
    args = parser.parse_args()

    google_api_key = os.environ.get("GOOGLE_MAPS_API_KEY")

    historical_stops, historical_summary = load_ga_stops(Path(args.historical_dir))
    live_stops, live_summary = load_ga_stops(Path(args.live_dir))

    depot = (historical_stops[0][0]['lat'], historical_stops[0][0]['lon'])

    fetcher = RouteGeometryFetcher(cache_file=args.geometry_cache, google_api_key=google_api_key)

    m = folium.Map(location=depot, zoom_start=11, tiles="OpenStreetMap")
    folium.Marker(depot, tooltip="Depot (UDD PMI)", icon=folium.Icon(color='red', icon='home')).add_to(m)

    if not args.skip_baseline:
        overrides = load_geocode_overrides(Path(args.overrides_file))
        pairs = build_baseline_pairs(overrides, args.droping_file)
        add_baseline_layer(m, depot, pairs, fetcher)

    add_ga_layer(m, "OSRM-GA (regular GA)", historical_stops, historical_summary, fetcher.fetch_osrm)

    if google_api_key:
        add_ga_layer(m, "Google-GA (GA + Maps API)", live_stops, live_summary, fetcher.fetch_google)
    else:
        logger.warning("GOOGLE_MAPS_API_KEY not set - drawing the Google-GA layer with OSRM road "
                        "geometry instead (stop order still reflects the live-traffic-optimized route).")
        add_ga_layer(m, "Google-GA (GA + Maps API)", live_stops, live_summary, fetcher.fetch_osrm)

    folium.LayerControl(collapsed=False).add_to(m)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(output_path))
    logger.info(f"Map saved to {output_path}")


if __name__ == "__main__":
    main()
