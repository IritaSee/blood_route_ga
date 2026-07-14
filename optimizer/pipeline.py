"""
Route optimization pipeline: extract historical data -> geocode -> build a
distance/duration matrix (via whichever router is passed in) -> run GA -> save
results. This is main.py from the old repo with every deep-learning step removed
- the DL time predictor and DL route selector were out of scope from the start
(the client asked for GA-only), so they aren't ported here at all.

The router and GA implementation are both injected, so the same pipeline serves
both entry points:
  - run_historical.py -> OSRMRouter (routing_osrm.py) - static historical routing
  - run_live.py        -> GoogleMapsRouter (routing_google_maps.py) - live traffic

See run_historical.py / run_live.py for the actual CLI entry points.
"""

import logging
import json
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd

from optimizer.data_extractor import DataExtractor
from optimizer import geocoder as geocoder_module
from optimizer.comparison_reporter import ComparisonReporter

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class OptimizationPipeline:
    """Orchestrates blood supply routing optimization: data -> geocoding ->
    routing matrix -> GA -> results. Router and GA class are injected so the
    same pipeline works for both the historical (OSRM) and live (Google Maps)
    entry points."""

    def __init__(self,
                 router,
                 ga_class,
                 pmi_file: str = "data/Data PMI.xlsx",
                 droping_file: str = "data/All Droping.xlsx",
                 output_dir: str = "results",
                 num_vehicles: int = 2,
                 vehicle_capacity: float = 100.0):
        """
        Args:
            router: An object exposing build_matrix(locations) -> (duration, distance)
                and get_router_mode() -> str. Either OSRMRouter or GoogleMapsRouter.
            ga_class: Either GeneticAlgorithm (genetic_algorithm.py, untouched) or
                GeneticAlgorithm (genetic_algorithm_optimized.py, reworked). Both
                share the same constructor/run()/get_best_solution_details() shape.
            pmi_file: Path to Data PMI.xlsx
            droping_file: Path to All Droping.xlsx
            output_dir: Where to write results
            num_vehicles: Fleet size
            vehicle_capacity: Per-vehicle capacity (units)
        """
        self.router = router
        self.ga_class = ga_class
        self.pmi_file = pmi_file
        self.droping_file = droping_file
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.num_vehicles = num_vehicles
        self.vehicle_capacity = vehicle_capacity

        self.extractor = DataExtractor(pmi_file, droping_file)
        self.geocoder = geocoder_module.Geocoder(cache_file=str(self.output_dir / "geocode_cache.db"))

        self.locations: Optional[List[Dict]] = None
        self.duration_matrix = None
        self.distance_matrix = None
        self.ga_results: Optional[Dict] = None

        logger.info(f"Pipeline initialized (router: {router.get_router_mode()})")

    def extract_data(self) -> Dict:
        """Extract data from historical Excel files."""
        logger.info("=== STEP 1: Extract Data ===")
        summary = self.extractor.summarize_data()

        logger.info(f"Facilities: {summary['num_facilities']}")
        logger.info(f"Trip records: {summary['num_trip_records']}")
        logger.info(f"Hospitals: {summary['num_hospitals']}")
        logger.info(f"Unique locations: {summary['num_unique_locations']}")
        logger.info(f"Distance range: {summary['distance_range_km']['min']:.1f} - "
                    f"{summary['distance_range_km']['max']:.1f} km "
                    f"(avg: {summary['distance_range_km']['mean']:.1f} km)")

        return summary

    def geocode_locations(self, locations: List[Dict]) -> List[Dict]:
        """Geocode facility locations."""
        logger.info("=== STEP 2: Geocode Locations ===")
        logger.info(f"Geocoding {len(locations)} locations...")

        geocoded = self.geocoder.geocode_batch(locations)

        success_count = sum(1 for loc in geocoded if loc.get('lat') is not None)
        logger.info(f"Successfully geocoded: {success_count}/{len(geocoded)}")

        if success_count < len(geocoded):
            failed = [loc['name'] for loc in geocoded if loc.get('lat') is None]
            logger.warning(f"Failed to geocode: {failed}")

        self.locations = geocoded
        return geocoded

    def build_matrices(self) -> tuple:
        """Build distance and duration matrices via the injected router."""
        logger.info("=== STEP 3: Build Distance/Time Matrices ===")

        if not self.locations:
            raise ValueError("Locations not geocoded")

        logger.info(f"Building matrices for {len(self.locations)} locations "
                    f"({self.router.get_router_mode()})...")

        duration_matrix, distance_matrix = self.router.build_matrix(self.locations)

        self.duration_matrix = duration_matrix
        self.distance_matrix = distance_matrix

        logger.info(f"Duration matrix shape: {duration_matrix.shape}")
        logger.info(f"Distance matrix shape: {distance_matrix.shape}")

        return duration_matrix, distance_matrix

    def extract_baseline(self, trip_data: pd.DataFrame) -> Dict:
        """Extract baseline metrics from historical trip data, for comparison
        against the GA result."""
        logger.info("=== STEP 4: Extract Baseline Metrics ===")

        if trip_data.empty:
            logger.warning("No trip history data available")
            return {}

        valid_trips = trip_data[trip_data['distance_km'].notna()].copy()
        if valid_trips.empty:
            return {}

        baseline = {
            'num_trips': len(valid_trips),
            'avg_distance_km': valid_trips['distance_km'].mean(),
            'avg_duration_hours': None,
            'avg_cost_idr': None,
            'on_time_percentage': 0,
        }

        if 'status' in valid_trips.columns:
            on_time = (valid_trips['status'].str.lower() == 'tepat waktu').sum()
            baseline['on_time_percentage'] = (on_time / len(valid_trips)) * 100

        try:
            valid_trips['trip_duration_hours'] = pd.to_timedelta(
                valid_trips['trip_duration']
            ).dt.total_seconds() / 3600
            baseline['avg_duration_hours'] = valid_trips['trip_duration_hours'].mean()
        except Exception:
            pass

        baseline['avg_cost_idr'] = valid_trips['distance_km'].mean() * (12750 / 9)

        logger.info("Baseline metrics:")
        logger.info(f"  Trips: {baseline['num_trips']}")
        logger.info(f"  Avg distance: {baseline['avg_distance_km']:.1f} km")
        logger.info(f"  Avg cost: {baseline['avg_cost_idr']:.0f} IDR")
        logger.info(f"  On-time: {baseline['on_time_percentage']:.1f}%")

        return baseline

    def optimize(self, population_size: int = 150, generations: int = 800,
                 quantities=None) -> Dict:
        """Run GA optimization using whichever GA class was injected."""
        logger.info("=== STEP 5: Genetic Algorithm Optimization ===")

        if self.duration_matrix is None or self.distance_matrix is None:
            raise ValueError("Matrices not built")

        num_customers = len(self.locations) - 1  # exclude depot

        logger.info(f"Starting GA: {num_customers} customers, {self.num_vehicles} vehicles")
        logger.info(f"Parameters: pop={population_size}, gen={generations}")

        ga_kwargs = dict(
            num_customers=num_customers,
            num_vehicles=self.num_vehicles,
            duration_matrix=self.duration_matrix,
            distance_matrix=self.distance_matrix,
            vehicle_capacity=self.vehicle_capacity,
            population_size=population_size,
            generations=generations,
            crossover_rate=0.8,
            mutation_rate=0.1,
            elite_size=int(population_size * 0.1),
        )
        # genetic_algorithm_optimized.GeneticAlgorithm additionally accepts
        # `quantities` to enforce capacity; genetic_algorithm.GeneticAlgorithm
        # does not, so only pass it if supported.
        if quantities is not None and 'quantities' in self.ga_class.__init__.__code__.co_varnames:
            ga_kwargs['quantities'] = quantities

        ga = self.ga_class(**ga_kwargs)

        ga.run()
        self.ga_results = ga.get_best_solution_details()

        if not self.ga_results:
            logger.warning("Optimization finished but no feasible solution was recorded")
            return {}

        logger.info("Optimization complete!")
        logger.info(f"Best makespan: {self.ga_results.get('makespan_s', 0) / 3600:.2f} hours")
        logger.info(f"Total cost: {self.ga_results.get('total_cost_idr', 0):.0f} IDR")
        logger.info(f"Total distance: {self.ga_results.get('total_distance_km', 0):.1f} km")

        return self.ga_results

    def generate_comparison_report(self, baseline: Dict) -> Optional[str]:
        """Write a GA-vs-historical-baseline comparison report."""
        if not self.ga_results:
            logger.warning("No GA results available for comparison report")
            return None

        reporter = ComparisonReporter(output_dir=str(self.output_dir))
        report_path = reporter.generate_text_report(baseline, self.ga_results)
        reporter.generate_json_report(baseline, self.ga_results)
        return report_path

    def save_results(self, baseline: Dict):
        """Save optimization results to JSON."""
        logger.info("=== STEP 6: Save Results ===")

        if not self.ga_results:
            logger.warning("No GA results to save")
            return

        results_file = self.output_dir / "ga_results.json"
        with open(results_file, 'w') as f:
            results_to_save = {
                'router_mode': self.router.get_router_mode(),
                'makespan_hours': self.ga_results['makespan_s'] / 3600,
                'total_time_hours': self.ga_results['total_time_s'] / 3600,
                'total_distance_km': self.ga_results['total_distance_km'],
                'total_cost_idr': self.ga_results['total_cost_idr'],
                'vehicle_distances_km': [d / 1000 for d in self.ga_results['vehicle_distances_m']],
                'vehicle_times_hours': [t / 3600 for t in self.ga_results['vehicle_times_s']],
                'vehicle_costs_idr': self.ga_results['vehicle_costs_idr'],
                'num_routes': self.ga_results['num_routes'],
            }
            json.dump(results_to_save, f, indent=2)

        logger.info(f"Results saved to {results_file}")

        if baseline:
            comparison = {
                'baseline_distance_km': baseline.get('avg_distance_km'),
                'ga_distance_km': self.ga_results['total_distance_km'],
                'distance_reduction_pct': (
                    (1 - self.ga_results['total_distance_km'] /
                     (baseline.get('avg_distance_km', 1) * self.ga_results['num_routes']))
                    * 100
                ) if baseline.get('avg_distance_km') else 0,
                'baseline_cost_idr': baseline.get('avg_cost_idr'),
                'ga_cost_idr': self.ga_results['total_cost_idr'],
                'cost_reduction_pct': (
                    (1 - self.ga_results['total_cost_idr'] /
                     (baseline.get('avg_cost_idr', 1) * self.ga_results['num_routes']))
                    * 100
                ) if baseline.get('avg_cost_idr') else 0,
            }

            comp_file = self.output_dir / "comparison.json"
            with open(comp_file, 'w') as f:
                json.dump(comparison, f, indent=2)

            logger.info(f"Comparison saved to {comp_file}")

    def run_full_pipeline(self, population_size: int = 150, generations: int = 800,
                           quantities=None) -> Dict:
        """Execute the full GA-only pipeline: extract -> geocode -> matrices ->
        baseline -> optimize -> compare -> save."""
        logger.info("\n" + "=" * 60)
        logger.info(f"BLOOD SUPPLY ROUTE OPTIMIZATION - {self.router.get_router_mode()}")
        logger.info("Malang Regency, Indonesia")
        logger.info("=" * 60 + "\n")

        summary = self.extract_data()
        locations = self.extractor.get_all_locations()
        geocoded = self.geocode_locations(locations)
        duration_matrix, distance_matrix = self.build_matrices()
        baseline = self.extract_baseline(summary['trip_history'])
        ga_results = self.optimize(population_size, generations, quantities=quantities)
        comparison_report_path = self.generate_comparison_report(baseline)
        self.save_results(baseline)

        logger.info("\n" + "=" * 60)
        logger.info("PIPELINE COMPLETE")
        logger.info("=" * 60 + "\n")

        return {
            'summary': summary,
            'locations': geocoded,
            'baseline': baseline,
            'ga_results': ga_results,
            'comparison_report': comparison_report_path,
        }
