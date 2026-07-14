"""
Google Maps Distance Matrix routing module - the "live feed" counterpart to
routing_osrm.py. Same public interface (build_matrix, get_router_mode) so it is a
drop-in swap for OSRMRouter in run_live.py.

What makes this "live" rather than a second static backend: every request sets
departure_time to now (or a caller-supplied future time) and reads back
`duration_in_traffic`, which factors in Google's real-time traffic conditions -
OSRM's public table API has no notion of traffic at all, it only knows the road
network. Falls back to plain `duration` if `duration_in_traffic` is unavailable
for a pair (e.g. transit mode, or Google couldn't compute it).

Because traffic conditions change minute to minute, caching is short-TTL
(GOOGLE_MAPS_CACHE_TTL_S, default 5 minutes) rather than the effectively-permanent
cache OSRM uses for static road-network data - the goal is to avoid re-hitting the
API for every GA fitness evaluation within a single run, not to serve stale
traffic data across runs.

Requires a Google Maps API key with the Distance Matrix API enabled, passed via
the GOOGLE_MAPS_API_KEY environment variable (see .env.example).
"""

import os
import sqlite3
import time
import logging
import json
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import numpy as np

import requests

logger = logging.getLogger(__name__)


class GoogleMapsRouterError(RuntimeError):
    """Raised when the router is misconfigured (e.g. missing API key)."""


class LiveRoutingCache:
    """SQLite cache for Google Maps distance/duration pairs, short-TTL because
    traffic conditions go stale quickly."""

    def __init__(self, cache_file: str = "routing_cache_live.db", ttl_s: int = 300):
        self.cache_file = Path(cache_file)
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)
        self.ttl_s = ttl_s
        self._init_db()

    def _init_db(self):
        try:
            with sqlite3.connect(self.cache_file) as conn:
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS pairs (
                        src_hash TEXT,
                        dst_hash TEXT,
                        duration_s REAL,
                        distance_m REAL,
                        timestamp REAL,
                        PRIMARY KEY (src_hash, dst_hash)
                    )
                ''')
                conn.commit()
        except Exception as e:
            logger.error(f"Error initializing live routing cache: {e}")

    def get_pair(self, src_hash: str, dst_hash: str) -> Optional[Tuple[float, float]]:
        try:
            with sqlite3.connect(self.cache_file) as conn:
                cursor = conn.execute(
                    '''SELECT duration_s, distance_m, timestamp FROM pairs
                       WHERE src_hash = ? AND dst_hash = ?''',
                    (src_hash, dst_hash)
                )
                row = cursor.fetchone()
                if row and (time.time() - row[2]) < self.ttl_s:
                    return (row[0], row[1])
            return None
        except Exception as e:
            logger.error(f"Error getting pair from live cache: {e}")
            return None

    def set_pair(self, src_hash: str, dst_hash: str, duration_s: float, distance_m: float) -> bool:
        try:
            with sqlite3.connect(self.cache_file) as conn:
                conn.execute('''
                    INSERT OR REPLACE INTO pairs
                    (src_hash, dst_hash, duration_s, distance_m, timestamp)
                    VALUES (?, ?, ?, ?, ?)
                ''', (src_hash, dst_hash, duration_s, distance_m, time.time()))
                conn.commit()
            return True
        except Exception as e:
            logger.error(f"Error caching live pair: {e}")
            return False


class GoogleMapsRouter:
    """Google Maps Distance Matrix routing with live traffic, batching, caching,
    and haversine fallback (matches OSRMRouter's fallback behavior)."""

    DISTANCE_MATRIX_URL = "https://maps.googleapis.com/maps/api/distancematrix/json"
    MIN_REQUEST_INTERVAL = 0.1  # Google's quota is generous; light throttling only
    MAX_ORIGINS_PER_REQUEST = 25   # Distance Matrix API element limits
    MAX_DESTINATIONS_PER_REQUEST = 25
    REQUEST_TIMEOUT = 30
    BACKOFF_INTERVALS = [1, 2, 4, 8, 16]
    MAX_RETRIES = 5

    def __init__(self, cache_file: str = "routing_cache_live.db",
                 api_key: Optional[str] = None,
                 use_live_traffic: bool = True,
                 traffic_model: str = "best_guess",
                 cache_ttl_s: int = 300,
                 fallback_speed_kmh: float = 40.0):
        """
        Args:
            cache_file: SQLite cache file path
            api_key: Google Maps API key. Falls back to GOOGLE_MAPS_API_KEY env var.
            use_live_traffic: If True, request duration_in_traffic with
                departure_time=now. If False, behaves like a static (no-traffic)
                Google Maps backend.
            traffic_model: "best_guess" | "pessimistic" | "optimistic"
                (Google Maps traffic_model parameter)
            cache_ttl_s: Cache TTL in seconds for pair lookups
            fallback_speed_kmh: Speed for haversine calculation if the API call fails
        """
        self.api_key = api_key or os.environ.get("GOOGLE_MAPS_API_KEY")
        if not self.api_key:
            raise GoogleMapsRouterError(
                "No Google Maps API key found. Set GOOGLE_MAPS_API_KEY in your "
                "environment (see .env.example) or pass api_key= explicitly."
            )

        self.cache = LiveRoutingCache(cache_file, ttl_s=cache_ttl_s)
        self.use_live_traffic = use_live_traffic
        self.traffic_model = traffic_model
        self.fallback_speed_kmh = fallback_speed_kmh
        self.last_request_time = 0.0

    def _rate_limit(self):
        elapsed = time.time() - self.last_request_time
        if elapsed < self.MIN_REQUEST_INTERVAL:
            time.sleep(self.MIN_REQUEST_INTERVAL - elapsed)
        self.last_request_time = time.time()

    def _haversine_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        from math import radians, cos, sin, asin, sqrt
        lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
        c = 2 * asin(sqrt(a))
        return c * 6371000

    def _get_location_hash(self, lat: float, lon: float) -> str:
        return f"{lat:.6f},{lon:.6f}"

    def _request_block(self, origins: List[Tuple[float, float]],
                        destinations: List[Tuple[float, float]]) -> Optional[Dict]:
        """Single Distance Matrix API call for one origins x destinations block."""
        params = {
            'origins': "|".join(f"{lat},{lon}" for lat, lon in origins),
            'destinations': "|".join(f"{lat},{lon}" for lat, lon in destinations),
            'mode': 'driving',
            'key': self.api_key,
        }
        if self.use_live_traffic:
            params['departure_time'] = 'now'
            params['traffic_model'] = self.traffic_model

        for attempt in range(self.MAX_RETRIES):
            try:
                self._rate_limit()
                response = requests.get(self.DISTANCE_MATRIX_URL, params=params,
                                         timeout=self.REQUEST_TIMEOUT)

                if response.status_code != 200:
                    logger.error(f"Google Maps HTTP error: {response.status_code}")
                    return None

                data = response.json()
                status = data.get('status')

                if status == 'OK':
                    return data
                elif status in ('OVER_QUERY_LIMIT', 'UNKNOWN_ERROR'):
                    if attempt < self.MAX_RETRIES - 1:
                        wait_time = self.BACKOFF_INTERVALS[min(attempt, len(self.BACKOFF_INTERVALS) - 1)]
                        logger.warning(f"Google Maps throttled ({status}), retrying in {wait_time}s")
                        time.sleep(wait_time)
                    else:
                        logger.error(f"Google Maps failed after {self.MAX_RETRIES} attempts ({status})")
                        return None
                else:
                    logger.error(f"Google Maps error: {status} - {data.get('error_message', '')}")
                    return None

            except requests.exceptions.RequestException as e:
                logger.error(f"Google Maps request failed: {e}")
                if attempt < self.MAX_RETRIES - 1:
                    wait_time = self.BACKOFF_INTERVALS[min(attempt, len(self.BACKOFF_INTERVALS) - 1)]
                    time.sleep(wait_time)
                else:
                    return None

        return None

    def build_matrix(self, locations: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build live-traffic-aware distance and duration matrices for locations.

        Args:
            locations: List of dicts with 'name', 'lat', 'lon'

        Returns:
            Tuple of (duration_matrix_seconds, distance_matrix_meters). Durations
            reflect current traffic conditions when use_live_traffic=True.
        """
        n = len(locations)
        duration_matrix = np.zeros((n, n))
        distance_matrix = np.zeros((n, n))

        valid_indices = [i for i, loc in enumerate(locations)
                          if loc.get('lat') is not None and loc.get('lon') is not None]
        for i, loc in enumerate(locations):
            if i not in valid_indices:
                logger.warning(f"Missing coordinates for {loc.get('name')}")
                duration_matrix[i, :] = np.inf
                distance_matrix[i, :] = np.inf

        if not valid_indices:
            logger.error("No valid coordinates for matrix building")
            return duration_matrix, distance_matrix

        logger.info(f"Building {n}x{n} live-traffic matrix via Google Maps "
                    f"({len(valid_indices)} valid coordinates, traffic={self.use_live_traffic})")

        # Resolve from cache first, collect the uncached pairs to fetch
        to_fetch_origins = set()
        to_fetch_destinations = set()
        for i in valid_indices:
            for j in valid_indices:
                if i == j:
                    duration_matrix[i, j] = 0
                    distance_matrix[i, j] = 0
                    continue
                src_hash = self._get_location_hash(locations[i]['lat'], locations[i]['lon'])
                dst_hash = self._get_location_hash(locations[j]['lat'], locations[j]['lon'])
                cached = self.cache.get_pair(src_hash, dst_hash)
                if cached:
                    duration_matrix[i, j], distance_matrix[i, j] = cached
                else:
                    to_fetch_origins.add(i)
                    to_fetch_destinations.add(j)

        fetch_origins = sorted(to_fetch_origins)
        fetch_destinations = sorted(to_fetch_destinations)

        for oi_start in range(0, len(fetch_origins), self.MAX_ORIGINS_PER_REQUEST):
            oi_end = min(oi_start + self.MAX_ORIGINS_PER_REQUEST, len(fetch_origins))
            origin_indices = fetch_origins[oi_start:oi_end]
            origin_coords = [(locations[i]['lat'], locations[i]['lon']) for i in origin_indices]

            for di_start in range(0, len(fetch_destinations), self.MAX_DESTINATIONS_PER_REQUEST):
                di_end = min(di_start + self.MAX_DESTINATIONS_PER_REQUEST, len(fetch_destinations))
                dest_indices = fetch_destinations[di_start:di_end]
                dest_coords = [(locations[j]['lat'], locations[j]['lon']) for j in dest_indices]

                response = self._request_block(origin_coords, dest_coords)

                if response:
                    rows = response.get('rows', [])
                    for local_i, global_i in enumerate(origin_indices):
                        if local_i >= len(rows):
                            continue
                        elements = rows[local_i].get('elements', [])
                        for local_j, global_j in enumerate(dest_indices):
                            if global_i == global_j or local_j >= len(elements):
                                continue
                            element = elements[local_j]
                            if element.get('status') != 'OK':
                                continue

                            duration_s = (
                                element.get('duration_in_traffic', {}).get('value')
                                if self.use_live_traffic else None
                            )
                            if duration_s is None:
                                duration_s = element.get('duration', {}).get('value')
                            distance_m = element.get('distance', {}).get('value')

                            if duration_s is None or distance_m is None:
                                continue

                            duration_matrix[global_i, global_j] = duration_s
                            distance_matrix[global_i, global_j] = distance_m

                            src_hash = self._get_location_hash(*origin_coords[local_i])
                            dst_hash = self._get_location_hash(*dest_coords[local_j])
                            self.cache.set_pair(src_hash, dst_hash, duration_s, distance_m)
                else:
                    logger.warning("Google Maps block failed, will fall back to haversine for those pairs")

        # Haversine fallback for anything still unfilled (API failure or missing element)
        for i in valid_indices:
            for j in valid_indices:
                if duration_matrix[i, j] == 0 and i != j:
                    dist_m = self._haversine_distance(
                        locations[i]['lat'], locations[i]['lon'],
                        locations[j]['lat'], locations[j]['lon']
                    )
                    distance_matrix[i, j] = dist_m
                    duration_matrix[i, j] = (dist_m / 1000.0) / self.fallback_speed_kmh * 3600

        logger.info(f"Live matrix complete: duration range "
                    f"{np.min(duration_matrix[duration_matrix > 0]):.0f}-"
                    f"{np.max(duration_matrix):.0f}s, "
                    f"distance range "
                    f"{np.min(distance_matrix[distance_matrix > 0]):.0f}-"
                    f"{np.max(distance_matrix):.0f}m")

        return duration_matrix, distance_matrix

    def get_router_mode(self) -> str:
        return "Google Maps (live traffic)" if self.use_live_traffic else "Google Maps (static)"
