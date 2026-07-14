"""
Geocoding module with Nominatim (OSM) support, caching, and rate limiting.
Converts facility addresses to GPS coordinates using OpenStreetMap Nominatim API.
"""

import sqlite3
import time
import logging
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import json
import hashlib

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
    """Nominatim geocoder with caching and rate limiting."""
    
    NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
    MIN_REQUEST_INTERVAL = 1.0  # 1 second minimum between requests
    REQUEST_TIMEOUT = 10
    
    def __init__(self, cache_file: str = "geocode_cache.db", 
                 user_agent: str = "BloodSupplyGA/0.1.0 (optimization@malang.local)"):
        """Initialize geocoder."""
        self.cache = GeocoderCache(cache_file)
        self.user_agent = user_agent
        self.last_request_time = 0
        self.request_count = 0
    
    def _rate_limit(self):
        """Enforce rate limiting (max 1 req/sec)."""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.MIN_REQUEST_INTERVAL:
            time.sleep(self.MIN_REQUEST_INTERVAL - elapsed)
        self.last_request_time = time.time()
    
    def geocode(self, address: str, city: str = "Malang", 
                country_code: str = "ID", retry: int = 3) -> Optional[Dict]:
        """
        Geocode an address using Nominatim with retries.
        
        Args:
            address: Address or facility name
            city: City (default: Malang)
            country_code: Country code (default: ID)
            retry: Number of retries on failure
        
        Returns:
            Dict with lat, lon, display_name, etc. or None if failed
        """
        # Check cache first
        query = f"{address}, {city}, {country_code}"
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
        
        # Make request
        for attempt in range(retry):
            try:
                self._rate_limit()
                
                params = {
                    'q': query,
                    'format': 'json',
                    'countrycodes': country_code,
                    'limit': 1,
                    'viewbox': f"{MALANG_BOUNDS['west']},{MALANG_BOUNDS['north']}" +
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
                    geocode_result = {
                        'lat': float(result['lat']),
                        'lon': float(result['lon']),
                        'display_name': result.get('display_name', ''),
                        'osm_id': result.get('osm_id'),
                        'osm_type': result.get('osm_type', ''),
                        'importance': float(result.get('importance', 0)),
                        'success': True,
                    }
                    self.cache.set(query, geocode_result)
                    logger.info(f"Geocoded: {query} -> ({geocode_result['lat']}, {geocode_result['lon']})")
                    return geocode_result
                else:
                    logger.warning(f"No results for: {query}")
                    self.cache.set(query, {'success': False})
                    return None
            
            except requests.exceptions.RequestException as e:
                logger.warning(f"Geocoding attempt {attempt + 1}/{retry} failed for {query}: {e}")
                if attempt < retry - 1:
                    backoff = 1.5 ** attempt  # Exponential backoff
                    time.sleep(backoff)
                else:
                    logger.error(f"Geocoding failed after {retry} attempts: {query}")
                    return None
        
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
            
            # Try full address first
            address = f"{loc['name']}, {loc.get('location_district', 'Malang')}"
            result = self.geocode(address)
            
            if not result:
                # Fallback: try just the name
                logger.debug(f"Retrying with just name: {loc['name']}")
                result = self.geocode(loc['name'])
            
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
