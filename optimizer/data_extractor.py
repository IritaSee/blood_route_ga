"""
Extract and parse data from Excel files (Data PMI.xlsx, All Droping.xlsx)
for GA optimization including facilities, distances, demands, and trip history.
"""

import pandas as pd
import numpy as np
import re
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class DataExtractor:
    """Extract Malang blood supply chain data from Excel files."""
    
    def __init__(self, pmi_file: str = "Data PMI.xlsx", droping_file: str = "All Droping.xlsx"):
        """Initialize data extractor."""
        self.pmi_file = Path(pmi_file)
        self.droping_file = Path(droping_file)

    def _is_non_location_name(self, name: str) -> bool:
        """Filter summary/number labels that are not physical locations."""
        n = str(name).strip()
        if not n:
            return True

        upper = n.upper()
        if upper in {"JUMLAH", "RATA-RATA"}:
            return True

        if re.fullmatch(r"\d+(\.\d+)?", n):
            return True

        if re.match(r"^[A-Z]\.\s", n):
            return True

        return False
        
    def extract_facilities(self) -> pd.DataFrame:
        """
        Extract unique facility locations from Jarak Pengiriman sheet.
        Returns DataFrame with columns: name, location, description
        """
        try:
            df = pd.read_excel(self.pmi_file, sheet_name='Jarak Pengiriman', header=None)
            
            # Skip header rows (typically rows 0-6 are metadata)
            # Row 5 has actual headers, data starts from row 7
            start_row = 7  # Data starts after headers
            
            facilities = []
            for idx in range(start_row, len(df)):
                row = df.iloc[idx]
                
                # Extract facility info - data is in columns 1,2,3,4
                facility_no = row.iloc[1]
                facility_name = row.iloc[2]
                location = row.iloc[3]
                distance = row.iloc[4]
                
                # Skip if name is NaN or empty
                if pd.isna(facility_name) or str(facility_name).strip() == '':
                    continue
                
                # Clean distance value (remove 'm' suffix if present)
                try:
                    distance_val = float(str(distance).replace('m', '').replace('Km', '').strip())
                except (ValueError, TypeError, AttributeError):
                    distance_val = None
                
                facilities.append({
                    'id': int(facility_no) if pd.notna(facility_no) else len(facilities),
                    'name': str(facility_name).strip(),
                    'location': str(location).strip() if pd.notna(location) else '',
                    'distance_from_pmi': distance_val,
                    'source': 'Jarak Pengiriman'
                })
            
            return pd.DataFrame(facilities)
        
        except Exception as e:
            logger.error(f"Error extracting facilities: {e}")
            return pd.DataFrame()
    
    def extract_trip_history(self) -> pd.DataFrame:
        """
        Extract historical trip data from Keterlambatan sheet.
        Returns DataFrame with columns: date, status, deliveries, distance, 
                                      departure_time, arrival_time, duration, delay_factor
        """
        try:
            df = pd.read_excel(self.pmi_file, sheet_name='Keterlambatan (tidak dipakai)', header=None)
            
            # Row 2 has headers
            header_row = 2
            
            trips = []
            for idx in range(header_row + 1, len(df)):
                row = df.iloc[idx]
                
                month = row.iloc[0]
                date = row.iloc[1]
                status = row.iloc[2]
                delay_factor = row.iloc[3]
                deliveries = row.iloc[4]
                distance = row.iloc[5]
                departure = row.iloc[6]
                arrival = row.iloc[7]
                duration = row.iloc[8]
                
                # Skip if date is NaN
                if pd.isna(date) or str(date).strip() == '':
                    continue
                
                # Try to clean and parse distance
                try:
                    if pd.notna(distance):
                        distance_val = float(str(distance).replace(',', '.').strip())
                    else:
                        distance_val = None
                except (ValueError, TypeError, AttributeError):
                    distance_val = None
                
                trips.append({
                    'month': month,
                    'date': pd.to_datetime(date, errors='coerce') if pd.notna(date) else None,
                    'status': str(status).strip() if pd.notna(status) else '',
                    'delay_factor': str(delay_factor).strip() if pd.notna(delay_factor) else '',
                    'deliveries': str(deliveries).strip() if pd.notna(deliveries) else '',
                    'distance_km': distance_val,
                    'departure_time': departure if pd.notna(departure) else None,
                    'arrival_time': arrival if pd.notna(arrival) else None,
                    'trip_duration': duration if pd.notna(duration) else None,
                })
            
            # Filter to valid trips with dates
            valid_trips = [t for t in trips if t['date'] is not None]
            return pd.DataFrame(valid_trips)
        
        except Exception as e:
            logger.error(f"Error extracting trip history: {e}")
            return pd.DataFrame()
    
    def extract_demand_by_hospital(self) -> pd.DataFrame:
        """
        Extract demand by hospital/facility from Permintaan Perwilayah 2024 sheet.
        Returns DataFrame with columns: hospital_name, location, monthly_demand (Jan-Dec)
        """
        try:
            df = pd.read_excel(self.pmi_file, sheet_name='Permintaan Perwilayah 2024', header=None)
            
            # Row 5 has headers: No, Nama RS, Lokasi, Bulan columns
            # Row 6 has months: Januari, Februari, ...
            header_row = 5
            month_row = 6
            start_data_row = 8
            
            # Extract month names
            months = []
            for col in range(3, min(15, len(df.columns))):
                month = df.iloc[month_row, col]
                if pd.notna(month) and str(month).strip() != '':
                    months.append(str(month).strip())
            
            demands = []
            for idx in range(start_data_row, len(df)):
                row = df.iloc[idx]
                
                hospital_no = row.iloc[0]
                hospital_name = row.iloc[1]
                location = row.iloc[2]
                
                # Skip if hospital name is NaN or empty or section header
                if (pd.isna(hospital_name) or str(hospital_name).strip() == '' or 
                    'Wilayah' in str(hospital_name)):
                    continue

                if self._is_non_location_name(hospital_name):
                    continue
                
                # Extract monthly demand values
                demand_dict = {
                    'id': int(hospital_no) if pd.notna(hospital_no) and str(hospital_no).replace('.', '', 1).isdigit() else len(demands),
                    'name': str(hospital_name).strip(),
                    'location': str(location).strip() if pd.notna(location) else '',
                }
                
                for i, month in enumerate(months):
                    col = 3 + i
                    demand_val = row.iloc[col] if col < len(row) else None
                    try:
                        demand_dict[month] = float(demand_val) if pd.notna(demand_val) and str(demand_val).replace('.', '', 1).replace(',', '', 1).isdigit() else 0.0
                    except (ValueError, TypeError):
                        demand_dict[month] = 0.0
                
                demands.append(demand_dict)
            
            return pd.DataFrame(demands)
        
        except Exception as e:
            logger.error(f"Error extracting demand: {e}")
            return pd.DataFrame()
    
    def get_all_locations(self) -> List[Dict]:
        """
        Get all unique facility locations combining facilities and hospitals.
        Returns list of location dictionaries for geocoding.
        """
        facilities = self.extract_facilities()
        demands = self.extract_demand_by_hospital()
        
        # Combine and deduplicate by name
        all_locs = []
        seen_names = set()
        
        for _, row in facilities.iterrows():
            name = row['name']
            if self._is_non_location_name(name):
                continue
            if name not in seen_names:
                all_locs.append({
                    'name': name,
                    'location_district': row.get('location', ''),
                    'type': 'facility'
                })
                seen_names.add(name)
        
        for _, row in demands.iterrows():
            name = row['name']
            if self._is_non_location_name(name):
                continue
            if name not in seen_names:
                all_locs.append({
                    'name': name,
                    'location_district': row.get('location', ''),
                    'type': 'hospital'
                })
                seen_names.add(name)
        
        # Add PMI blood bank as depot
        if 'UDD PMI Kabupaten Malang' not in seen_names:
            all_locs.insert(0, {
                'name': 'UDD PMI Kabupaten Malang',
                'location_district': 'Malang',
                'type': 'blood_bank'
            })
        
        return all_locs
    
    def summarize_data(self) -> Dict:
        """Get data summary for validation."""
        facilities = self.extract_facilities()
        trips = self.extract_trip_history()
        demands = self.extract_demand_by_hospital()
        locations = self.get_all_locations()
        
        return {
            'num_facilities': len(facilities),
            'num_trip_records': len(trips),
            'num_hospitals': len(demands),
            'num_unique_locations': len(locations),
            'date_range': {
                'start': trips['date'].min() if 'date' in trips.columns else None,
                'end': trips['date'].max() if 'date' in trips.columns else None,
            },
            'distance_range_km': {
                'min': trips['distance_km'].min() if 'distance_km' in trips.columns else None,
                'max': trips['distance_km'].max() if 'distance_km' in trips.columns else None,
                'mean': trips['distance_km'].mean() if 'distance_km' in trips.columns else None,
            },
            'facilities': facilities,
            'trip_history': trips,
            'demand': demands,
            'locations': locations,
        }
