"""
Entry point 2: GA route optimization with live traffic feed (Google Maps API).

Requires GOOGLE_MAPS_API_KEY in your environment or a .env file
(see .env.example). Distance Matrix API must be enabled on that key.

Usage:
    python run_live.py
    python run_live.py --ga optimized
    python run_live.py --no-live-traffic   # static Google Maps, no traffic
"""

import argparse
import os

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from optimizer.pipeline import OptimizationPipeline
from optimizer.routing_google_maps import GoogleMapsRouter, GoogleMapsRouterError
from optimizer.genetic_algorithm import GeneticAlgorithm as GeneticAlgorithmBaseline
from optimizer.genetic_algorithm_optimized import GeneticAlgorithm as GeneticAlgorithmOptimized


def main():
    parser = argparse.ArgumentParser(description="GA route optimization with live Google Maps traffic")
    parser.add_argument("--ga", choices=["baseline", "optimized"], default="optimized",
                         help="Which GA implementation to run (default: optimized)")
    parser.add_argument("--population", type=int, default=150)
    parser.add_argument("--generations", type=int, default=800)
    parser.add_argument("--vehicles", type=int, default=2)
    parser.add_argument("--capacity", type=float, default=100.0)
    parser.add_argument("--no-live-traffic", action="store_true",
                         help="Use Google Maps without live traffic (static duration)")
    parser.add_argument("--traffic-model", choices=["best_guess", "pessimistic", "optimistic"],
                         default="best_guess")
    parser.add_argument("--cache-ttl", type=int, default=300,
                         help="Seconds to cache a live routing pair before refetching (default: 300)")
    parser.add_argument("--output-dir", default="results/live")
    args = parser.parse_args()

    ga_class = GeneticAlgorithmOptimized if args.ga == "optimized" else GeneticAlgorithmBaseline

    try:
        router = GoogleMapsRouter(
            cache_file=f"{args.output_dir}/routing_cache_live.db",
            use_live_traffic=not args.no_live_traffic,
            traffic_model=args.traffic_model,
            cache_ttl_s=args.cache_ttl,
        )
    except GoogleMapsRouterError as e:
        raise SystemExit(f"{e}\n\nSet GOOGLE_MAPS_API_KEY in your environment or .env file.")

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
    )


if __name__ == "__main__":
    main()
