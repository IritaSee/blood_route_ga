"""
Geocoding module with Google Maps URL parsing as primary lookup.
Falls back to Google Places API and then Nominatim when needed.
"""

import sqlite3
import time
import logging
import os
import re
import csv
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import json
import hashlib
from urllib.parse import quote_plus

import requests

logger = logging.getLogger(__name__)

# Malang Regency bounding box (approximate)
MALANG_BOUNDS = {
    'south': -8.5,
    'north': -7.3,
    'west': 112.2,
    'east': 112.9
}


class GeocoderCache:
    """SQLite-based cache for geocoding results."""
    
    def __init__(self, cache_file: str = "geocode_cache.db"):
        """Initialize geocoding cache."""
        self.cache_file = Path(cache_file)
        self._init_db()
    
    def _init_db(self):
        """Create cache database if it doesn't exist."""
        try:
            with sqlite3.connect(self.cache_file) as conn:
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS geocodes (
                        query_hash TEXT PRIMARY KEY,
                        query TEXT,
                        lat REAL,
                        lon REAL,
                        display_name TEXT,
                        osm_id INTEGER,
                        osm_type TEXT,
                        importance REAL,
                        address_json TEXT,
                        timestamp REAL,
                        success BOOLEAN
                    )
                ''')
                conn.commit()
        except Exception as e:
            logger.error(f"Error initializing geocode cache: {e}")
    
    def _hash_query(self, query: str) -> str:
        """Create hash of normalized query."""
        normalized = query.strip().lower()
        return hashlib.md5(normalized.encode()).hexdigest()
    
    def get(self, query: str) -> Optional[Dict]:
        """Get cached geocoding result."""
        try:
            query_hash = self._hash_query(query)
            with sqlite3.connect(self.cache_file) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute(
                    'SELECT * FROM geocodes WHERE query_hash = ?',
                    (query_hash,)
                )
                row = cursor.fetchone()
                
                if row:
                    return dict(row)
            return None
        except Exception as e:
            logger.error(f"Error getting from cache: {e}")
            return None
    
    def set(self, query: str, result: Dict) -> bool:
        """Cache geocoding result."""
        try:
            query_hash = self._hash_query(query)
            with sqlite3.connect(self.cache_file) as conn:
                conn.execute('''
                    INSERT OR REPLACE INTO geocodes
                    (query_hash, query, lat, lon, display_name, osm_id, osm_type, 
                     importance, address_json, timestamp, success)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    query_hash,
                    query,
                    result.get('lat'),
                    result.get('lon'),
                    result.get('display_name', ''),
                    result.get('osm_id'),
                    result.get('osm_type', ''),
                    result.get('importance'),
                    json.dumps(result.get('address', {})),
                    time.time(),
                    result.get('success', False)
                ))
                conn.commit()
            return True
        except Exception as e:
            logger.error(f"Error caching result: {e}")
            return False


class Geocoder:
    """Geocoder with Google Maps URL parsing as primary strategy."""
    
    NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
    PLACES_FIND_URL = "https://maps.googleapis.com/maps/api/place/findplacefromtext/json"
    GOOGLE_MAPS_SEARCH_URL = "https://www.google.com/maps/search/?api=1&query="
    MIN_REQUEST_INTERVAL = 1.0  # 1 second minimum between requests
    REQUEST_TIMEOUT = 10
    
    def __init__(self, cache_file: str = "geocode_cache.db", 
                 user_agent: str = "BloodSupplyGA/0.1.0 (optimization@malang.local)"):
        """Initialize geocoder."""
        self.cache = GeocoderCache(cache_file)
        self.user_agent = user_agent
        self.google_api_key = os.environ.get("GOOGLE_MAPS_API_KEY")
        self.last_request_time = 0
        self.request_count = 0
        self.manual_overrides = self._load_manual_overrides()
    
    def _rate_limit(self):
        """Enforce rate limiting (max 1 req/sec)."""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.MIN_REQUEST_INTERVAL:
            time.sleep(self.MIN_REQUEST_INTERVAL - elapsed)
        self.last_request_time = time.time()
    
    def _build_query(self, address: str, city: Optional[str], country_code: str) -> str:
        address = str(address).strip()
        parts = [address]
        if city:
            parts.append(str(city).strip())
        parts.append(country_code)
        return ", ".join(p for p in parts if p)

    def _normalize(self, value: Optional[str]) -> str:
        return re.sub(r"\s+", " ", str(value or "").strip().lower())

    def _parse_float(self, value: Optional[str]) -> Optional[float]:
        if value is None:
            return None
        text = str(value).strip()
        if not text:
            return None
        try:
            return float(text)
        except ValueError:
            return None

    def _candidate_override_files(self) -> List[Path]:
        env_file = os.environ.get("GEOCODE_OVERRIDE_FILE")
        if env_file:
            return [Path(env_file)]

        return [
            Path("data/geocode_overrides.csv"),
            Path("data/geocode_gap_report.csv"),
            Path("data/geocode_gap_report (1).csv"),
        ]

    def _load_manual_overrides(self) -> Dict[Tuple[str, str], Dict]:
        for csv_path in self._candidate_override_files():
            if not csv_path.exists():
                continue

            overrides: Dict[Tuple[str, str], Dict] = {}
            try:
                with csv_path.open("r", encoding="utf-8", newline="") as fh:
                    reader = csv.DictReader(fh)
                    for row in reader:
                        name = str(row.get("name", "")).strip()
                        district = str(row.get("location_district", "")).strip()
                        lat = self._parse_float(row.get("lat"))
                        lon = self._parse_float(row.get("lon"))

                        if not name or lat is None or lon is None:
                            continue

                        key = (self._normalize(name), self._normalize(district))
                        overrides[key] = {
                            'lat': lat,
                            'lon': lon,
                            'display_name': name,
                            'osm_id': None,
                            'osm_type': 'manual_csv',
                            'importance': 1.0,
                            'success': True,
                        }

                        if district:
                            fallback_key = (self._normalize(name), "")
                            overrides.setdefault(fallback_key, overrides[key])

                logger.info(f"Loaded {len(overrides)} manual geocode overrides from {csv_path}")
                return overrides
            except Exception as e:
                logger.warning(f"Failed to read manual geocode overrides from {csv_path}: {e}")

        return {}

    def _lookup_manual_override(self, name: str, district: Optional[str]) -> Optional[Dict]:
        if not self.manual_overrides:
            return None

        key_exact = (self._normalize(name), self._normalize(district))
        if key_exact in self.manual_overrides:
            return self.manual_overrides[key_exact]

        key_name_only = (self._normalize(name), "")
        return self.manual_overrides.get(key_name_only)

    def _geocode_with_places(self, query: str, retry: int = 3) -> Optional[Dict]:
        """Use Google Places Find Place from Text to get coordinates."""
        if not self.google_api_key:
            return None

        params = {
            'input': query,
            'inputtype': 'textquery',
            'fields': 'place_id,name,formatted_address,geometry/location',
            'key': self.google_api_key,
        }

        for attempt in range(retry):
            try:
                self._rate_limit()
                response = requests.get(
                    self.PLACES_FIND_URL,
                    params=params,
                    timeout=self.REQUEST_TIMEOUT,
                )
                response.raise_for_status()

                data = response.json()
                status = data.get('status')
                candidates = data.get('candidates', [])

                if status == 'OK' and candidates:
                    candidate = candidates[0]
                    location = candidate.get('geometry', {}).get('location', {})
                    lat = location.get('lat')
                    lon = location.get('lng')

                    if lat is None or lon is None:
                        return None

                    return {
                        'lat': float(lat),
                        'lon': float(lon),
                        'display_name': candidate.get('formatted_address', candidate.get('name', '')),
                        'osm_id': candidate.get('place_id'),
                        'osm_type': 'google_place',
                        'importance': 1.0,
                        'success': True,
                    }

                if status in ('ZERO_RESULTS', 'NOT_FOUND'):
                    return None

                logger.warning(f"Google Places error for {query}: {status} - {data.get('error_message', '')}")
                if status in ('OVER_QUERY_LIMIT', 'UNKNOWN_ERROR') and attempt < retry - 1:
                    time.sleep(1.5 ** attempt)
                    continue
                return None

            except requests.exceptions.RequestException as e:
                logger.warning(f"Google Places attempt {attempt + 1}/{retry} failed for {query}: {e}")
                if attempt < retry - 1:
                    time.sleep(1.5 ** attempt)
                else:
                    return None

        return None

    def _geocode_with_nominatim(self, query: str, country_code: str, retry: int = 3) -> Optional[Dict]:
        """Fallback geocoding path using Nominatim."""
        for attempt in range(retry):
            try:
                self._rate_limit()

                params = {
                    'q': query,
                    'format': 'json',
                    'countrycodes': country_code,
                    'limit': 1,
                    'viewbox': f"{MALANG_BOUNDS['west']},{MALANG_BOUNDS['north']}"
                               f",{MALANG_BOUNDS['east']},{MALANG_BOUNDS['south']}",
                    'bounded': 1,
                }

                headers = {'User-Agent': self.user_agent}

                response = requests.get(
                    self.NOMINATIM_URL,
                    params=params,
                    headers=headers,
                    timeout=self.REQUEST_TIMEOUT
                )
                response.raise_for_status()

                results = response.json()

                if results:
                    result = results[0]
                    return {
                        'lat': float(result['lat']),
                        'lon': float(result['lon']),
                        'display_name': result.get('display_name', ''),
                        'osm_id': result.get('osm_id'),
                        'osm_type': result.get('osm_type', ''),
                        'importance': float(result.get('importance', 0)),
                        'success': True,
                    }

                return None

            except requests.exceptions.RequestException as e:
                logger.warning(f"Nominatim attempt {attempt + 1}/{retry} failed for {query}: {e}")
                if attempt < retry - 1:
                    backoff = 1.5 ** attempt
                    time.sleep(backoff)
                else:
                    logger.error(f"Nominatim failed after {retry} attempts: {query}")
                    return None

        return None

    def _geocode_with_google_maps_search_url(self, query: str, retry: int = 3) -> Optional[Dict]:
        """Fallback that uses Google Maps search URL and parses @lat,lon from final URL.

        This is useful when API responses are unavailable but Google Maps web search
        still resolves a place URL containing coordinates.
        """
        headers = {
            'User-Agent': self.user_agent,
            'Accept-Language': 'en-US,en;q=0.9',
        }

        encoded_query = quote_plus(query)
        url = f"{self.GOOGLE_MAPS_SEARCH_URL}{encoded_query}"

        for attempt in range(retry):
            try:
                self._rate_limit()
                response = requests.get(
                    url,
                    headers=headers,
                    timeout=self.REQUEST_TIMEOUT,
                    allow_redirects=True,
                )
                response.raise_for_status()

                final_url = str(response.url or "")

                # Common Google Maps coordinate pattern: .../@-7.9617511,112.680274,16.94z
                match = re.search(r"/@(-?\d+\.\d+),(-?\d+\.\d+)", final_url)
                if not match:
                    # Fallback pattern sometimes appears as !3dLAT!4dLON in URL path/query.
                    match_alt = re.search(r"!3d(-?\d+\.\d+)!4d(-?\d+\.\d+)", final_url)
                    if not match_alt:
                        return None
                    lat = float(match_alt.group(1))
                    lon = float(match_alt.group(2))
                else:
                    lat = float(match.group(1))
                    lon = float(match.group(2))

                return {
                    'lat': lat,
                    'lon': lon,
                    'display_name': query,
                    'osm_id': None,
                    'osm_type': 'google_maps_search_url',
                    'importance': 1.0,
                    'success': True,
                }

            except requests.exceptions.RequestException as e:
                logger.warning(f"Google Maps search URL attempt {attempt + 1}/{retry} failed for {query}: {e}")
                if attempt < retry - 1:
                    time.sleep(1.5 ** attempt)
                else:
                    return None

        return None

    def geocode(self, address: str, city: Optional[str] = None,
                country_code: str = "ID", retry: int = 3) -> Optional[Dict]:
        """
        Geocode an address with retries.
        
        Args:
            address: Address or facility name
            city: City (default: Malang)
            country_code: Country code (default: ID)
            retry: Number of retries on failure
        
        Returns:
            Dict with lat, lon, display_name, etc. or None if failed
        """
        # Check cache first
        query = self._build_query(address, city, country_code)
        cached = self.cache.get(query)
        if cached and cached.get('success'):
            logger.debug(f"Cache hit: {query}")
            return {
                'lat': cached['lat'],
                'lon': cached['lon'],
                'display_name': cached['display_name'],
                'osm_id': cached['osm_id'],
                'osm_type': cached['osm_type'],
                'importance': cached['importance'],
            }
        
        geocode_result = self._geocode_with_google_maps_search_url(query, retry=retry)
        if not geocode_result:
            geocode_result = self._geocode_with_places(query, retry=retry)
        if not geocode_result:
            geocode_result = self._geocode_with_nominatim(query, country_code, retry=retry)

        if geocode_result:
            self.cache.set(query, geocode_result)
            logger.info(f"Geocoded: {query} -> ({geocode_result['lat']}, {geocode_result['lon']})")
            return geocode_result

        logger.warning(f"No results for: {query}")
        self.cache.set(query, {'success': False})
        return None
    
    def geocode_batch(self, locations: List[Dict]) -> List[Dict]:
        """
        Geocode multiple locations with progress tracking.
        
        Args:
            locations: List of dicts with 'name' and 'location_district'
        
        Returns:
            List of dicts with added 'lat' and 'lon' fields
        """
        geocoded = []
        for i, loc in enumerate(locations):
            logger.info(f"Geocoding {i + 1}/{len(locations)}: {loc['name']}")

            manual = self._lookup_manual_override(loc['name'], loc.get('location_district'))
            if manual:
                loc_copy = loc.copy()
                loc_copy.update(manual)
                geocoded.append(loc_copy)
                logger.debug(f"Manual override used: {loc['name']}")
                continue
            
            # Try full address first
            district = loc.get('location_district')
            address = f"{loc['name']}, {district}" if district else loc['name']
            result = self.geocode(address)
            
            if not result:
                # Fallback: try just the name
                logger.debug(f"Retrying with just name: {loc['name']}")
                result = self.geocode(loc['name'], city="Malang")
            
            if result:
                loc_copy = loc.copy()
                loc_copy.update(result)
                geocoded.append(loc_copy)
                logger.debug(f"Success: {loc['name']}")
            else:
                logger.warning(f"Geocoding failed: {loc['name']}")
                # Add placeholder
                loc_copy = loc.copy()
                loc_copy['lat'] = None
                loc_copy['lon'] = None
                geocoded.append(loc_copy)
        
        return geocoded
    
    def get_cache_stats(self) -> Dict:
        """Get cache statistics."""
        try:
            with sqlite3.connect(self.cache.cache_file) as conn:
                cursor = conn.execute('SELECT COUNT(*) as total, SUM(success) as successful FROM geocodes')
                row = cursor.fetchone()
                return {
                    'total_cached': row[0],
                    'successful': row[1],
                    'failed': row[0] - (row[1] or 0),
                }
        except Exception as e:
            logger.error(f"Error getting cache stats: {e}")
            return {}
