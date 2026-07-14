"""
Extract baseline metrics from historical trip data for comparison.
"""

import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class BaselineExtractor:
    """Extract historical trip metrics from All Droping.xlsx."""
    
    def __init__(self, file_path: str = "All Droping.xlsx"):
        """Initialize baseline extractor."""
        self.file_path = Path(file_path)
        if not self.file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        self.trip_data = None
        self._load_data()
    
    def _load_data(self):
        """Load and clean trip data."""
        logger.info(f"Loading trip data from {self.file_path}")
        
        # Read with correct header row
        df = pd.read_excel(
            self.file_path,
            sheet_name="Keterlambatan & Waktu Trip",
            header=1
        )
        
        # Filter to rows with actual trip data (has date and distance)
        df = df[df['Tanggal Pengiriman'].notna()].copy()
        df = df[df['Jarak (km)'].notna()].copy()
        
        # Clean distance column
        df['Jarak (km)'] = pd.to_numeric(df['Jarak (km)'], errors='coerce')
        df = df[df['Jarak (km)'] > 0]
        
        # Parse time columns
        df['Tanggal Pengiriman'] = pd.to_datetime(
            df['Tanggal Pengiriman'],
            errors='coerce'
        )
        
        # Clean duration and lateness
        df['convert Durasi (Menit)'] = pd.to_numeric(
            df['convert Durasi (Menit)'],
            errors='coerce'
        )
        df['Convert Waktu Terlambat (Menit)'] = pd.to_numeric(
            df['Convert Waktu Terlambat (Menit)'],
            errors='coerce'
        )
        
        # Fill NaN lateness with 0 (on time)
        df['Convert Waktu Terlambat (Menit)'] = df[
            'Convert Waktu Terlambat (Menit)'
        ].fillna(0)
        
        # Status: on-time or late
        df['is_late'] = df['Convert Waktu Terlambat (Menit)'] > 0
        
        self.trip_data = df
        logger.info(f"Loaded {len(self.trip_data)} valid trips")
    
    def get_overall_baseline(self) -> Dict:
        """Get overall baseline metrics from all trips."""
        if self.trip_data is None or self.trip_data.empty:
            logger.warning("No trip data available")
            return {}
        
        trips = self.trip_data
        
        # Basic statistics
        metrics = {
            'num_trips': len(trips),
            'avg_distance_km': trips['Jarak (km)'].mean(),
            'min_distance_km': trips['Jarak (km)'].min(),
            'max_distance_km': trips['Jarak (km)'].max(),
            'total_distance_km': trips['Jarak (km)'].sum(),
            'avg_duration_hours': (
                trips['convert Durasi (Menit)'].mean() / 60
                if trips['convert Durasi (Menit)'].notna().any()
                else None
            ),
            'avg_lateness_minutes': trips[
                'Convert Waktu Terlambat (Menit)'
            ].mean(),
            'on_time_count': (~trips['is_late']).sum(),
            'late_count': trips['is_late'].sum(),
            'on_time_percentage': (
                (~trips['is_late']).sum() / len(trips) * 100
            ) if len(trips) > 0 else 0,
            'late_percentage': (
                trips['is_late'].sum() / len(trips) * 100
            ) if len(trips) > 0 else 0,
        }
        
        # Cost calculation: 12,750 IDR/L, ~9 km/L
        fuel_price_per_liter = 12750
        fuel_efficiency_km_per_liter = 9
        cost_per_km = fuel_price_per_liter / fuel_efficiency_km_per_liter
        metrics['cost_per_km_idr'] = cost_per_km
        metrics['avg_cost_per_trip_idr'] = (
            metrics['avg_distance_km'] * cost_per_km
        )
        metrics['total_cost_idr'] = (
            metrics['total_distance_km'] * cost_per_km
        )
        
        return metrics
    
    def get_trip_details(self) -> pd.DataFrame:
        """Return trip dataframe for detailed analysis."""
        return self.trip_data[[
            'Tanggal Pengiriman',
            'Tujuan 1',
            'Jarak (km)',
            'convert Durasi (Menit)',
            'Convert Waktu Terlambat (Menit)',
            'is_late',
            'Status',
        ]].copy() if self.trip_data is not None else pd.DataFrame()
    
    def get_monthly_baseline(self) -> Dict[str, Dict]:
        """Get baseline metrics by month."""
        if self.trip_data is None or self.trip_data.empty:
            return {}
        
        trips = self.trip_data.copy()
        trips['month'] = trips['Tanggal Pengiriman'].dt.to_period('M')
        
        monthly = {}
        for month, group in trips.groupby('month'):
            if len(group) > 0:
                monthly[str(month)] = {
                    'num_trips': len(group),
                    'avg_distance_km': group['Jarak (km)'].mean(),
                    'total_distance_km': group['Jarak (km)'].sum(),
                    'avg_duration_hours': (
                        group['convert Durasi (Menit)'].mean() / 60
                        if group['convert Durasi (Menit)'].notna().any()
                        else None
                    ),
                    'on_time_percentage': (
                        (~group['is_late']).sum() / len(group) * 100
                    ),
                    'late_percentage': (
                        group['is_late'].sum() / len(group) * 100
                    ),
                    'avg_lateness_minutes': group[
                        'Convert Waktu Terlambat (Menit)'
                    ].mean(),
                }
        
        return monthly
    
    def get_destination_baseline(self) -> Dict[str, Dict]:
        """Get baseline metrics by destination."""
        if self.trip_data is None or self.trip_data.empty:
            return {}
        
        trips = self.trip_data.copy()
        dest_stats = {}
        
        for destination in trips['Tujuan 1'].unique():
            if pd.isna(destination):
                continue
            
            dest_trips = trips[trips['Tujuan 1'] == destination]
            if len(dest_trips) > 0:
                dest_stats[str(destination)] = {
                    'num_trips': len(dest_trips),
                    'avg_distance_km': dest_trips['Jarak (km)'].mean(),
                    'avg_duration_hours': (
                        dest_trips['convert Durasi (Menit)'].mean() / 60
                        if dest_trips['convert Durasi (Menit)'].notna().any()
                        else None
                    ),
                    'on_time_percentage': (
                        (~dest_trips['is_late']).sum() / len(dest_trips) * 100
                    ),
                    'avg_lateness_minutes': dest_trips[
                        'Convert Waktu Terlambat (Menit)'
                    ].mean(),
                }
        
        return dest_stats
    
    def compare_ga_to_baseline(self, ga_results: Dict) -> Dict:
        """
        Compare GA results to historical baseline.
        
        Args:
            ga_results: Dict with GA metrics (total_distance_km, total_cost_idr, etc.)
        
        Returns:
            Comparison dict with improvements/deviations
        """
        baseline = self.get_overall_baseline()
        
        if not baseline:
            logger.warning("Cannot compare: no baseline metrics")
            return {}
        
        # Extract metrics
        baseline_distance = baseline['total_distance_km']
        baseline_cost = baseline['total_cost_idr']
        ga_distance = ga_results.get('total_distance_km', 0)
        ga_cost = ga_results.get('total_cost_idr', 0)
        
        # Calculate improvements
        # Note: GA is for optimized routes, baseline is total historical
        # For fair comparison, scale baseline by number of routes in GA
        num_routes = ga_results.get('num_routes', 1)
        
        # Comparison: what if we ran GA on one "optimized batch"?
        # vs average baseline per trip * similar number of stops
        comparison = {
            'baseline_metrics': baseline,
            'ga_metrics': {
                'total_distance_km': ga_distance,
                'total_cost_idr': ga_cost,
                'makespan_hours': ga_results.get('makespan_hours', 0),
                'total_time_hours': ga_results.get('total_time_hours', 0),
                'num_routes': num_routes,
            },
            'improvements': {
                'distance_reduction_km': baseline_distance - ga_distance,
                'distance_reduction_pct': (
                    ((baseline_distance - ga_distance) / baseline_distance * 100)
                    if baseline_distance > 0 else 0
                ),
                'cost_reduction_idr': baseline_cost - ga_cost,
                'cost_reduction_pct': (
                    ((baseline_cost - ga_cost) / baseline_cost * 100)
                    if baseline_cost > 0 else 0
                ),
            },
            'summary': {
                'baseline_on_time_pct': baseline['on_time_percentage'],
                'baseline_avg_lateness_min': baseline['avg_lateness_minutes'],
                'ga_num_routes': num_routes,
                'ga_makespan_hours': ga_results.get('makespan_hours', 0),
            }
        }
        
        return comparison
    
    def summarize(self, include_monthly: bool = True, 
                  include_destinations: bool = True) -> Dict:
        """Get comprehensive summary of baseline data."""
        summary = {
            'overall': self.get_overall_baseline(),
        }
        
        if include_monthly:
            summary['monthly'] = self.get_monthly_baseline()
        
        if include_destinations:
            summary['by_destination'] = self.get_destination_baseline()
        
        return summary
