"""
Self-contained correctness checks for both GA implementations, using mock
distance/duration matrices (no Excel data, no network) so these run fast and
in any environment.
"""

import numpy as np

from optimizer.genetic_algorithm import GeneticAlgorithm as BaselineGA
from optimizer.genetic_algorithm_optimized import GeneticAlgorithm as OptimizedGA


def _mock_matrices(num_locations: int, seed: int = 42):
    """Symmetric random distance/duration matrices with zero diagonal."""
    rng = np.random.default_rng(seed)
    coords = rng.uniform(0, 50_000, size=(num_locations, 2))  # meters, flat plane
    distance_matrix = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
    np.fill_diagonal(distance_matrix, 0)
    duration_matrix = distance_matrix / (40_000 / 3600)  # 40 km/h flat speed
    return duration_matrix, distance_matrix


def _run_ga(ga_class, num_customers=10, num_vehicles=2, generations=30, **kwargs):
    duration_matrix, distance_matrix = _mock_matrices(num_customers + 1)
    ga = ga_class(
        num_customers=num_customers,
        num_vehicles=num_vehicles,
        duration_matrix=duration_matrix,
        distance_matrix=distance_matrix,
        vehicle_capacity=100.0,
        population_size=30,
        generations=generations,
        elite_size=3,
        **kwargs,
    )
    ga.run()
    return ga


def test_baseline_ga_runs_and_covers_all_customers():
    ga = _run_ga(BaselineGA)
    details = ga.get_best_solution_details()
    assert details, "baseline GA should return a best solution"

    all_customers = sorted(c for route in details['routes'] for c in route)
    assert all_customers == list(range(10)), "every customer must appear exactly once"


def test_optimized_ga_runs_and_covers_all_customers():
    ga = _run_ga(OptimizedGA)
    details = ga.get_best_solution_details()
    assert details, "optimized GA should return a best solution"

    all_customers = sorted(c for route in details['routes'] for c in route)
    assert all_customers == list(range(10)), "every customer must appear exactly once"


def test_optimized_ga_improves_over_generations():
    ga = _run_ga(OptimizedGA, generations=60)
    history = ga.fitness_history
    assert history[-1]['best_fitness'] <= history[0]['best_fitness']


def test_optimized_ga_penalizes_capacity_violation():
    """With all demand deliberately overloaded onto too little total capacity,
    the best solution's fitness should still include a nonzero penalty term -
    this is the constraint genetic_algorithm.py has no way to express."""
    num_customers = 8
    duration_matrix, distance_matrix = _mock_matrices(num_customers + 1)
    quantities = np.array([30.0] * num_customers)  # 240 total demand

    ga = OptimizedGA(
        num_customers=num_customers,
        num_vehicles=2,
        duration_matrix=duration_matrix,
        distance_matrix=distance_matrix,
        vehicle_capacity=50.0,  # 2 vehicles x 50 = 100 < 240 total demand: infeasible
        quantities=quantities,
        population_size=20,
        generations=20,
        elite_size=2,
    )
    ga.run()
    details = ga.get_best_solution_details()
    assert details['penalty'] > 0, "overloaded routes must be penalized"


def test_crossover_preserves_customer_membership():
    """genetic_algorithm.py's original crossover only ever reassigns vehicle
    IDs; the optimized crossover operates on ordered sequences directly and
    must still end up with exactly one occurrence of each customer post-repair."""
    from optimizer.genetic_algorithm_optimized import Chromosome, GeneticAlgorithm

    num_customers = 12
    duration_matrix, distance_matrix = _mock_matrices(num_customers + 1)
    ga = GeneticAlgorithm(
        num_customers=num_customers,
        num_vehicles=3,
        duration_matrix=duration_matrix,
        distance_matrix=distance_matrix,
    )

    parent1 = Chromosome.random_init(num_customers, 3)
    parent2 = Chromosome.random_init(num_customers, 3)
    child = ga.crossover(parent1, parent2)

    all_customers = sorted(c for route in child.get_routes() for c in route)
    assert all_customers == list(range(num_customers))
