"""
OSRM routing module with batching, caching, and rate limiting.
Computes distance and time matrices from geocoded facility coordinates.
"""

import sqlite3
import time
import logging
import json
import pickle
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import numpy as np

import requests

logger = logging.getLogger(__name__)


class RoutingCache:
    """SQLite cache for OSRM distance/time matrices."""
    
    def __init__(self, cache_file: str = "routing_cache.db"):
        """Initialize routing cache."""
        self.cache_file = Path(cache_file)
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
    
    def _init_db(self):
        """Create cache database."""
        try:
            with sqlite3.connect(self.cache_file) as conn:
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS matrices (
                        profile TEXT,
                        dataset_tag TEXT,
                        data_json TEXT,
                        location_hashes TEXT,
                        timestamp REAL,
                        PRIMARY KEY (profile, dataset_tag)
                    )
                ''')
                
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS pairs (
                        profile TEXT,
                        src_hash TEXT,
                        dst_hash TEXT,
                        duration_s REAL,
                        distance_m REAL,
                        timestamp REAL,
                        PRIMARY KEY (profile, src_hash, dst_hash)
                    )
                ''')
                
                conn.commit()
        except Exception as e:
            logger.error(f"Error initializing routing cache: {e}")
    
    def get_matrix(self, profile: str, dataset_tag: str) -> Optional[Dict]:
        """Get cached distance/time matrix."""
        try:
            with sqlite3.connect(self.cache_file) as conn:
                cursor = conn.execute(
                    '''SELECT data_json FROM matrices 
                       WHERE profile = ? AND dataset_tag = ?''',
                    (profile, dataset_tag)
                )
                row = cursor.fetchone()
                if row:
                    return json.loads(row[0])
            return None
        except Exception as e:
            logger.error(f"Error getting matrix from cache: {e}")
            return None
    
    def set_matrix(self, profile: str, dataset_tag: str, 
                   matrix_data: Dict, location_hashes: List[str]) -> bool:
        """Cache distance/time matrix."""
        try:
            with sqlite3.connect(self.cache_file) as conn:
                conn.execute('''
                    INSERT OR REPLACE INTO matrices
                    (profile, dataset_tag, data_json, location_hashes, timestamp)
                    VALUES (?, ?, ?, ?, ?)
                ''', (
                    profile,
                    dataset_tag,
                    json.dumps(matrix_data),
                    json.dumps(location_hashes),
                    time.time()
                ))
                conn.commit()
            return True
        except Exception as e:
            logger.error(f"Error caching matrix: {e}")
            return False
    
    def get_pair(self, profile: str, src_hash: str, dst_hash: str) -> Optional[Tuple[float, float]]:
        """Get cached distance/duration for a location pair."""
        try:
            with sqlite3.connect(self.cache_file) as conn:
                cursor = conn.execute(
                    '''SELECT duration_s, distance_m FROM pairs 
                       WHERE profile = ? AND src_hash = ? AND dst_hash = ?''',
                    (profile, src_hash, dst_hash)
                )
                row = cursor.fetchone()
                if row:
                    return (row[0], row[1])
            return None
        except Exception as e:
            logger.error(f"Error getting pair from cache: {e}")
            return None
    
    def set_pair(self, profile: str, src_hash: str, dst_hash: str, 
                 duration_s: float, distance_m: float) -> bool:
        """Cache distance/duration for a location pair."""
        try:
            with sqlite3.connect(self.cache_file) as conn:
                conn.execute('''
                    INSERT OR REPLACE INTO pairs
                    (profile, src_hash, dst_hash, duration_s, distance_m, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    profile, src_hash, dst_hash, duration_s, distance_m, time.time()
                ))
                conn.commit()
            return True
        except Exception as e:
            logger.error(f"Error caching pair: {e}")
            return False


class OSRMRouter:
    """OSRM routing with batching, caching, and fallback."""
    
    OSRM_URL = "http://router.project-osrm.org/table/v1/car"
    MIN_REQUEST_INTERVAL = 1.0  # 1 sec minimum between requests
    MAX_COORDINATES_PER_REQUEST = 100
    MAX_CELLS_PER_REQUEST = 10000
    REQUEST_TIMEOUT = 30
    BACKOFF_INTERVALS = [1, 2, 4, 8, 16, 32]  # Exponential backoff
    MAX_RETRIES = 5
    
    def __init__(self, cache_file: str = "routing_cache.db", 
                 use_osrm: bool = True, fallback_speed_kmh: float = 40.0):
        """
        Initialize OSRM router.
        
        Args:
            cache_file: SQLite cache file path
            use_osrm: If True, use OSRM; if False, use haversine fallback
            fallback_speed_kmh: Speed for haversine calculation when OSRM unavailable
        """
        self.cache = RoutingCache(cache_file)
        self.use_osrm = use_osrm
        self.fallback_speed_kmh = fallback_speed_kmh
        self.last_request_time = 0
        self.request_count = 0
    
    def _rate_limit(self):
        """Enforce rate limiting (max 1 req/sec)."""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.MIN_REQUEST_INTERVAL:
            time.sleep(self.MIN_REQUEST_INTERVAL - elapsed)
        self.last_request_time = time.time()
    
    def _haversine_distance(self, lat1: float, lon1: float, 
                           lat2: float, lon2: float) -> float:
        """Calculate haversine distance in meters."""
        from math import radians, cos, sin, asin, sqrt
        
        lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
        c = 2 * asin(sqrt(a))
        r = 6371000  # Earth's radius in meters
        return c * r
    
    def _get_location_hash(self, lat: float, lon: float) -> str:
        """Create hash for location (rounded to 1e-6 precision)."""
        return f"{lat:.6f},{lon:.6f}"
    
    def _osrm_request(self, coordinates: List[Tuple[float, float]]) -> Optional[Dict]:
        """
        Make OSRM table request with error handling.
        
        Args:
            coordinates: List of (lat, lon) tuples
        
        Returns:
            OSRM response JSON or None if failed
        """
        # Check request size
        n = len(coordinates)
        cells = n * n
        if cells > self.MAX_CELLS_PER_REQUEST:
            logger.error(f"Too many cells: {cells} > {self.MAX_CELLS_PER_REQUEST}")
            return None
        
        # Build request
        coord_str = ";".join([f"{lon},{lat}" for lat, lon in coordinates])
        url = f"{self.OSRM_URL}/{coord_str}"
        
        params = {
            'annotations': 'distance,duration',
        }
        
        # Retry loop
        for attempt in range(self.MAX_RETRIES):
            try:
                self._rate_limit()
                
                logger.debug(f"OSRM request {attempt + 1}: {len(coordinates)} coordinates")
                response = requests.get(
                    url,
                    params=params,
                    timeout=self.REQUEST_TIMEOUT
                )
                
                if response.status_code == 200:
                    data = response.json()
                    if data.get('code') == 'Ok':
                        return data
                    else:
                        logger.warning(f"OSRM error: {data.get('code')} - {data.get('message')}")
                        return None
                
                elif response.status_code in [429, 503, 504]:
                    # Throttle/server error - retry with backoff
                    if attempt < self.MAX_RETRIES - 1:
                        wait_time = self.BACKOFF_INTERVALS[min(attempt, len(self.BACKOFF_INTERVALS) - 1)]
                        logger.warning(f"OSRM throttled (code {response.status_code}), "
                                     f"retrying in {wait_time}s")
                        time.sleep(wait_time)
                    else:
                        logger.error(f"OSRM failed after {self.MAX_RETRIES} attempts")
                        return None
                else:
                    logger.error(f"OSRM HTTP error: {response.status_code}")
                    return None
            
            except requests.exceptions.RequestException as e:
                logger.error(f"OSRM request failed: {e}")
                if attempt < self.MAX_RETRIES - 1:
                    wait_time = self.BACKOFF_INTERVALS[min(attempt, len(self.BACKOFF_INTERVALS) - 1)]
                    time.sleep(wait_time)
                else:
                    return None
        
        return None
    
    def build_matrix(self, locations: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build distance and duration matrices for locations.
        
        Args:
            locations: List of dicts with 'name', 'lat', 'lon'
        
        Returns:
            Tuple of (duration_matrix_seconds, distance_matrix_meters)
        """
        n = len(locations)
        duration_matrix = np.zeros((n, n))
        distance_matrix = np.zeros((n, n))
        
        # Validate coordinates
        coordinates = []
        valid_indices = []
        for i, loc in enumerate(locations):
            if loc.get('lat') is None or loc.get('lon') is None:
                logger.warning(f"Missing coordinates for {loc.get('name')}")
                duration_matrix[i, :] = np.inf
                distance_matrix[i, :] = np.inf
            else:
                coordinates.append((loc['lat'], loc['lon']))
                valid_indices.append(i)
        
        if not coordinates:
            logger.error("No valid coordinates for matrix building")
            return duration_matrix, distance_matrix
        
        # Use OSRM if enabled, otherwise use haversine
        if self.use_osrm:
            logger.info(f"Building {n}x{n} matrix via OSRM ({len(coordinates)} valid coordinates)")
            
            # Batch requests if needed
            batch_size = min(self.MAX_COORDINATES_PER_REQUEST, n)
            
            for i_start in range(0, len(coordinates), batch_size):
                i_end = min(i_start + batch_size, len(coordinates))
                batch_coords = coordinates[i_start:i_end]
                batch_indices = valid_indices[i_start:i_end]
                
                logger.debug(f"Processing batch {i_start//batch_size + 1}: "
                           f"{len(batch_coords)} coordinates")
                
                response = self._osrm_request(batch_coords)
                
                if response:
                    durations = np.array(response.get('durations', []))
                    distances = np.array(response.get('distances', []))
                    
                    # Fill into full matrix
                    for local_i, global_i in enumerate(batch_indices):
                        for local_j, global_j in enumerate(batch_indices):
                            duration_matrix[global_i, global_j] = durations[local_i, local_j]
                            distance_matrix[global_i, global_j] = distances[local_i, local_j]
                else:
                    logger.warning(f"OSRM batch failed, using fallback for batch {i_start//batch_size}")
        
        # Fill remaining with haversine (fallback)
        for i in valid_indices:
            for j in valid_indices:
                if duration_matrix[i, j] == 0 and i != j:
                    dist_m = self._haversine_distance(
                        locations[i]['lat'], locations[i]['lon'],
                        locations[j]['lat'], locations[j]['lon']
                    )
                    distance_matrix[i, j] = dist_m
                    duration_matrix[i, j] = (dist_m / 1000.0) / self.fallback_speed_kmh * 3600
        
        logger.info(f"Matrix complete: duration range "
                  f"{np.min(duration_matrix[duration_matrix > 0]):.0f}-"
                  f"{np.max(duration_matrix):.0f}s, "
                  f"distance range "
                  f"{np.min(distance_matrix[distance_matrix > 0]):.0f}-"
                  f"{np.max(distance_matrix):.0f}m")
        
        return duration_matrix, distance_matrix
    
    def get_router_mode(self) -> str:
        """Get current routing mode."""
        return "OSRM" if self.use_osrm else "Haversine"
