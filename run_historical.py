"""
Entry point 1: GA route optimization from historical data (OSRM routing).

Usage:
    python run_historical.py
    python run_historical.py --ga optimized
    python run_historical.py --population 200 --generations 500 --no-osrm
"""

import argparse

from optimizer.pipeline import OptimizationPipeline
from optimizer.routing_osrm import OSRMRouter
from optimizer.genetic_algorithm import GeneticAlgorithm as GeneticAlgorithmBaseline
from optimizer.genetic_algorithm_optimized import GeneticAlgorithm as GeneticAlgorithmOptimized


def main():
    parser = argparse.ArgumentParser(description="GA route optimization from historical data")
    parser.add_argument("--ga", choices=["baseline", "optimized"], default="optimized",
                         help="Which GA implementation to run (default: optimized)")
    parser.add_argument("--population", type=int, default=150)
    parser.add_argument("--generations", type=int, default=800)
    parser.add_argument("--vehicles", type=int, default=2)
    parser.add_argument("--capacity", type=float, default=100.0)
    parser.add_argument("--no-osrm", action="store_true",
                         help="Use haversine distance instead of OSRM (no network needed)")
    parser.add_argument("--output-dir", default="results/historical")
    parser.add_argument("--seed", type=int, default=None,
                         help="RNG seed for a reproducible GA run (default: unseeded)")
    args = parser.parse_args()

    ga_class = GeneticAlgorithmOptimized if args.ga == "optimized" else GeneticAlgorithmBaseline

    router = OSRMRouter(
        cache_file=f"{args.output_dir}/routing_cache.db",
        use_osrm=not args.no_osrm,
    )

    pipeline = OptimizationPipeline(
        router=router,
        ga_class=ga_class,
        output_dir=args.output_dir,
        num_vehicles=args.vehicles,
        vehicle_capacity=args.capacity,
    )

    pipeline.run_full_pipeline(
        population_size=args.population,
        generations=args.generations,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
