"""
Road-following route geometry ("breadcrumbs") for visualization.

optimizer/routing_osrm.py and optimizer/routing_google_maps.py both build a
distance/duration *matrix* (OSRM's /table endpoint, Google's Distance Matrix API) -
that's all the GA needs for optimization, but neither returns the actual path shape.
Drawing a route on a map needs a different endpoint family: OSRM's /route service,
or Google's Directions API. This module wraps those, given an ordered list of stops
(as visited by a GA route, or a depot<->destination pair for the historical
baseline), and returns the road-snapped polyline as a list of (lat, lon) points.

Same caching approach as the existing routing modules (sqlite, keyed by a hash of
the request) so re-running the visualizer doesn't re-fetch geometry that's already
been resolved.
"""

import hashlib
import json
import logging
import sqlite3
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests

logger = logging.getLogger(__name__)

Point = Tuple[float, float]  # (lat, lon)


class RouteGeometryError(RuntimeError):
    """Raised when geometry can't be resolved for a route (after retries)."""


class RouteGeometryCache:
    """SQLite cache for resolved route geometries, keyed by provider + ordered stops."""

    def __init__(self, cache_file: str = "route_geometry_cache.db"):
        self.cache_file = Path(cache_file)
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        try:
            with sqlite3.connect(self.cache_file) as conn:
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS geometries (
                        provider TEXT,
                        stops_hash TEXT,
                        geometry_json TEXT,
                        timestamp REAL,
                        PRIMARY KEY (provider, stops_hash)
                    )
                ''')
                conn.commit()
        except Exception as e:
            logger.error(f"Error initializing route geometry cache: {e}")

    def _hash_stops(self, stops: List[Point]) -> str:
        normalized = json.dumps([[round(lat, 6), round(lon, 6)] for lat, lon in stops])
        return hashlib.md5(normalized.encode()).hexdigest()

    def get(self, provider: str, stops: List[Point]) -> Optional[List[Point]]:
        try:
            with sqlite3.connect(self.cache_file) as conn:
                cursor = conn.execute(
                    'SELECT geometry_json FROM geometries WHERE provider = ? AND stops_hash = ?',
                    (provider, self._hash_stops(stops))
                )
                row = cursor.fetchone()
                if row:
                    return [tuple(p) for p in json.loads(row[0])]
            return None
        except Exception as e:
            logger.error(f"Error reading route geometry cache: {e}")
            return None

    def set(self, provider: str, stops: List[Point], geometry: List[Point]):
        try:
            with sqlite3.connect(self.cache_file) as conn:
                conn.execute('''
                    INSERT OR REPLACE INTO geometries (provider, stops_hash, geometry_json, timestamp)
                    VALUES (?, ?, ?, ?)
                ''', (provider, self._hash_stops(stops), json.dumps(geometry), time.time()))
                conn.commit()
        except Exception as e:
            logger.error(f"Error writing route geometry cache: {e}")


def _decode_google_polyline(encoded: str) -> List[Point]:
    """Decode a Google-encoded polyline string into (lat, lon) points."""
    index, lat, lng = 0, 0, 0
    coordinates: List[Point] = []
    changes = {'latitude': 0, 'longitude': 0}

    while index < len(encoded):
        for unit in ('latitude', 'longitude'):
            shift, result = 0, 0
            while True:
                byte = ord(encoded[index]) - 63
                index += 1
                result |= (byte & 0x1f) << shift
                shift += 5
                if byte < 0x20:
                    break
            changes[unit] = ~(result >> 1) if (result & 1) else (result >> 1)

        lat += changes['latitude']
        lng += changes['longitude']
        coordinates.append((lat / 1e5, lng / 1e5))

    return coordinates


class RouteGeometryFetcher:
    """Fetches road-snapped route geometry via OSRM's /route service or Google's
    Directions API, for an ordered list of (lat, lon) stops."""

    OSRM_ROUTE_URL = "http://router.project-osrm.org/route/v1/driving"
    OSRM_MIN_REQUEST_INTERVAL = 1.0
    GOOGLE_DIRECTIONS_URL = "https://maps.googleapis.com/maps/api/directions/json"
    GOOGLE_MIN_REQUEST_INTERVAL = 0.1
    # Google allows up to 25 waypoints per request (origin + destination + 23 via).
    GOOGLE_MAX_INTERMEDIATE_WAYPOINTS = 23
    REQUEST_TIMEOUT = 30
    BACKOFF_INTERVALS = [1, 2, 4, 8, 16]
    MAX_RETRIES = 5

    def __init__(self, cache_file: str = "route_geometry_cache.db",
                 google_api_key: Optional[str] = None):
        self.cache = RouteGeometryCache(cache_file)
        self.google_api_key = google_api_key
        self._last_osrm_request = 0.0
        self._last_google_request = 0.0

    def _rate_limit(self, last_attr: str, min_interval: float):
        last = getattr(self, last_attr)
        elapsed = time.time() - last
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)
        setattr(self, last_attr, time.time())

    def fetch_osrm(self, stops: List[Point]) -> List[Point]:
        """Road-snapped geometry for an ordered list of stops via OSRM's public
        /route demo server. Returns the stops unchanged (straight lines) if fewer
        than 2 stops are given."""
        if len(stops) < 2:
            return list(stops)

        cached = self.cache.get('osrm', stops)
        if cached is not None:
            return cached

        coord_str = ";".join(f"{lon},{lat}" for lat, lon in stops)
        url = f"{self.OSRM_ROUTE_URL}/{coord_str}"
        params = {'overview': 'full', 'geometries': 'geojson'}

        for attempt in range(self.MAX_RETRIES):
            try:
                self._rate_limit('_last_osrm_request', self.OSRM_MIN_REQUEST_INTERVAL)
                response = requests.get(url, params=params, timeout=self.REQUEST_TIMEOUT)

                if response.status_code == 200:
                    data = response.json()
                    if data.get('code') == 'Ok' and data.get('routes'):
                        coords = data['routes'][0]['geometry']['coordinates']
                        geometry = [(lat, lon) for lon, lat in coords]
                        self.cache.set('osrm', stops, geometry)
                        return geometry
                    logger.warning(f"OSRM route error: {data.get('code')} - {data.get('message')}")
                    return list(stops)

                if response.status_code in (429, 503, 504) and attempt < self.MAX_RETRIES - 1:
                    wait = self.BACKOFF_INTERVALS[min(attempt, len(self.BACKOFF_INTERVALS) - 1)]
                    logger.warning(f"OSRM route throttled ({response.status_code}), retrying in {wait}s")
                    time.sleep(wait)
                    continue

                logger.error(f"OSRM route HTTP error: {response.status_code}")
                return list(stops)

            except requests.exceptions.RequestException as e:
                logger.error(f"OSRM route request failed: {e}")
                if attempt < self.MAX_RETRIES - 1:
                    wait = self.BACKOFF_INTERVALS[min(attempt, len(self.BACKOFF_INTERVALS) - 1)]
                    time.sleep(wait)
                else:
                    return list(stops)

        return list(stops)

    def fetch_google(self, stops: List[Point]) -> List[Point]:
        """Road-snapped, traffic-informed geometry for an ordered list of stops via
        Google's Directions API. Chunks into <=25-waypoint requests (Google's cap)
        and concatenates leg geometries, so continuity is preserved across chunks.
        If a chunk's Directions request fails (e.g. the API key doesn't have
        Directions enabled, only Distance Matrix), that chunk falls back to OSRM's
        /route geometry rather than a straight line - still road-snapped, just not
        Google's live-traffic-aware path."""
        if len(stops) < 2:
            return list(stops)
        if not self.google_api_key:
            raise RouteGeometryError("No Google Maps API key configured for Directions requests.")

        cached = self.cache.get('google', stops)
        if cached is not None:
            return cached

        geometry: List[Point] = []
        chunk_span = self.GOOGLE_MAX_INTERMEDIATE_WAYPOINTS + 1  # stops covered per chunk (excl. shared boundary)
        start = 0
        while start < len(stops) - 1:
            end = min(start + chunk_span, len(stops) - 1)
            chunk = stops[start:end + 1]
            leg_geometry = self._fetch_google_chunk(chunk)
            if geometry and leg_geometry:
                leg_geometry = leg_geometry[1:]  # drop duplicate boundary point
            geometry.extend(leg_geometry)
            start = end

        if geometry:
            self.cache.set('google', stops, geometry)
        return geometry or list(stops)

    def _fetch_google_chunk(self, stops: List[Point]) -> List[Point]:
        origin = f"{stops[0][0]},{stops[0][1]}"
        destination = f"{stops[-1][0]},{stops[-1][1]}"
        params = {
            'origin': origin,
            'destination': destination,
            'mode': 'driving',
            'key': self.google_api_key,
        }
        if len(stops) > 2:
            params['waypoints'] = "|".join(f"{lat},{lon}" for lat, lon in stops[1:-1])

        for attempt in range(self.MAX_RETRIES):
            try:
                self._rate_limit('_last_google_request', self.GOOGLE_MIN_REQUEST_INTERVAL)
                response = requests.get(self.GOOGLE_DIRECTIONS_URL, params=params,
                                         timeout=self.REQUEST_TIMEOUT)

                if response.status_code != 200:
                    logger.error(f"Google Directions HTTP error: {response.status_code}")
                    return self.fetch_osrm(stops)

                data = response.json()
                status = data.get('status')

                if status == 'OK':
                    points: List[Point] = []
                    for leg in data['routes'][0]['legs']:
                        for step in leg['steps']:
                            decoded = _decode_google_polyline(step['polyline']['points'])
                            if points and decoded:
                                decoded = decoded[1:]
                            points.extend(decoded)
                    return points

                if status in ('OVER_QUERY_LIMIT', 'UNKNOWN_ERROR') and attempt < self.MAX_RETRIES - 1:
                    wait = self.BACKOFF_INTERVALS[min(attempt, len(self.BACKOFF_INTERVALS) - 1)]
                    logger.warning(f"Google Directions throttled ({status}), retrying in {wait}s")
                    time.sleep(wait)
                    continue

                logger.error(f"Google Directions error: {status} - {data.get('error_message', '')}")
                return self.fetch_osrm(stops)

            except requests.exceptions.RequestException as e:
                logger.error(f"Google Directions request failed: {e}")
                if attempt < self.MAX_RETRIES - 1:
                    wait = self.BACKOFF_INTERVALS[min(attempt, len(self.BACKOFF_INTERVALS) - 1)]
                    time.sleep(wait)
                else:
                    return self.fetch_osrm(stops)

        return self.fetch_osrm(stops)
