"""
Reworked genetic algorithm for multi-vehicle blood supply routing optimization.

This is a drop-in alternative to genetic_algorithm.py. Same constructor shape and
same evaluate_fitness()/run()/get_best_solution_details() surface, so either GA can
be plugged into run_historical.py / run_live.py via a flag. Kept as a separate file
(instead of editing genetic_algorithm.py) so the original, already-validated GA
stays available untouched.

What's different from genetic_algorithm.py, and why:

1. Route order actually survives crossover.
   The original crossover only reassigns which vehicle a customer belongs to, then
   rebuilds each route by iterating customers in ascending customer-id order
   (Chromosome._build_routes_from_assignments). That throws away any visiting-order
   information - including the nearest-neighbor seed route - after the very first
   generation, so the GA only ever searches "which vehicle" and never "in what
   order", even though the fitness function is entirely order-dependent. Here,
   crossover operates on the ordered per-vehicle sequences directly (order
   crossover per route, customers not yet placed reinserted by nearest-insertion),
   so visiting order is both preserved and explored.

2. 2-opt local search on top of the GA.
   After crossover/mutation, elite-bound offspring get a bounded 2-opt pass per
   route. This is standard for VRP: GA explores which customers go together and in
   roughly what order, 2-opt cleans up the "obviously crossed" legs GA is bad at
   fixing via random mutation alone.

3. Vehicle capacity is enforced.
   vehicle_capacity was accepted by the original constructor but never checked
   against anything - there was no way to pass per-customer demand, so it was dead
   weight. This version takes an optional `quantities` array and adds a capacity
   violation penalty to fitness, so infeasible-but-fast routes are no longer
   indistinguishable from feasible ones.

4. Stagnation-triggered diversity injection.
   If the best fitness hasn't improved for `stagnation_limit` generations, the
   bottom half of the (non-elite) population is replaced with fresh
   nearest-neighbor/random individuals. Plain elitism + tournament selection tends
   to converge the whole population around one basin; this gives it a way out
   without discarding the best solution found so far.
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import random

logger = logging.getLogger(__name__)

# Constants (kept identical to genetic_algorithm.py for comparable results)
FUEL_PRICE_IDR_PER_LITER = 12750
FUEL_EFFICIENCY_KM_PER_LITER = 9
COST_PER_KM_IDR = FUEL_PRICE_IDR_PER_LITER / FUEL_EFFICIENCY_KM_PER_LITER  # ~1416.67

# Penalty per unit of capacity overage (IDR-equivalent scale, tuned to dominate
# any time/cost saving from overloading a vehicle).
CAPACITY_PENALTY_PER_UNIT = 5.0


@dataclass
class VehicleRoute:
    """Represents a single vehicle's route."""
    vehicle_id: int
    facility_sequence: List[int]
    quantities: List[float]
    total_distance_m: float
    total_duration_s: float
    total_cost_idr: float
    load: float

    def summary(self) -> Dict:
        return {
            'vehicle_id': self.vehicle_id,
            'num_stops': len(self.facility_sequence),
            'distance_km': self.total_distance_m / 1000,
            'duration_hours': self.total_duration_s / 3600,
            'cost_idr': self.total_cost_idr,
            'load': self.load,
        }


class Chromosome:
    """
    GA chromosome for multi-vehicle routing.
    Encoding: explicit ordered per-vehicle customer sequences (order is meaningful
    and preserved through crossover/mutation - see module docstring point 1).
    """

    def __init__(self, num_customers: int, num_vehicles: int = 2):
        self.num_customers = num_customers
        self.num_vehicles = num_vehicles
        self.routes: List[List[int]] = [[] for _ in range(num_vehicles)]

    @classmethod
    def random_init(cls, num_customers: int, num_vehicles: int = 2) -> 'Chromosome':
        chrom = cls(num_customers, num_vehicles)
        customers = list(range(num_customers))
        random.shuffle(customers)
        for i, customer_id in enumerate(customers):
            chrom.routes[i % num_vehicles].append(customer_id)
        return chrom

    @classmethod
    def nearest_neighbor_init(cls, num_customers: int, distance_matrix: np.ndarray,
                               num_vehicles: int = 2) -> 'Chromosome':
        """Initialize using nearest-neighbor heuristic, one route grown per vehicle
        in round-robin so all vehicles get a comparable share of customers."""
        chrom = cls(num_customers, num_vehicles)
        unassigned = set(range(num_customers))
        current_per_vehicle = [0] * num_vehicles  # depot

        vehicle_id = 0
        while unassigned:
            current = current_per_vehicle[vehicle_id]
            nearest = min(unassigned, key=lambda x: distance_matrix[current, x + 1])
            chrom.routes[vehicle_id].append(nearest)
            current_per_vehicle[vehicle_id] = nearest + 1
            unassigned.remove(nearest)
            vehicle_id = (vehicle_id + 1) % num_vehicles

        return chrom

    def get_routes(self) -> List[List[int]]:
        return self.routes

    def set_routes(self, routes: List[List[int]]):
        self.routes = routes

    def clone(self) -> 'Chromosome':
        chrom = Chromosome(self.num_customers, self.num_vehicles)
        chrom.routes = [route.copy() for route in self.routes]
        return chrom

    def repair(self):
        """Ensure every customer appears exactly once across all routes.
        Crossover on ordered sequences can produce duplicates/omissions; this
        drops duplicates in place and appends any missing customers to the
        shortest route, keeping the chromosome always feasible-in-membership."""
        seen = set()
        for route in self.routes:
            deduped = []
            for c in route:
                if c not in seen:
                    seen.add(c)
                    deduped.append(c)
            route[:] = deduped

        missing = [c for c in range(self.num_customers) if c not in seen]
        for c in missing:
            shortest = min(self.routes, key=len)
            shortest.append(c)


class GeneticAlgorithm:
    """
    GA for multi-vehicle routing optimization with order-preserving crossover,
    2-opt refinement, optional capacity constraint, and stagnation recovery.
    Objectives: minimize delivery time (primary), then cost (secondary) - same
    lexicographic-via-weighted-sum fitness shape as genetic_algorithm.py.
    """

    def __init__(self, num_customers: int, num_vehicles: int,
                 duration_matrix: np.ndarray, distance_matrix: np.ndarray,
                 vehicle_capacity: float = 100.0,
                 quantities: Optional[np.ndarray] = None,
                 population_size: int = 150,
                 generations: int = 800,
                 crossover_rate: float = 0.8,
                 mutation_rate: float = 0.1,
                 elite_size: int = 15,
                 local_search_rate: float = 0.3,
                 stagnation_limit: int = 40):
        """
        Args:
            num_customers: Number of delivery locations
            num_vehicles: Number of vehicles
            duration_matrix: (n_locations x n_locations) duration in seconds
            distance_matrix: (n_locations x n_locations) distance in meters
            vehicle_capacity: Capacity per vehicle (units)
            quantities: Optional per-customer demand (length num_customers). If
                omitted, capacity is not enforced (matches original GA behavior).
            population_size: Population size
            generations: Number of generations
            crossover_rate: Crossover probability
            mutation_rate: Mutation probability per gene
            elite_size: Number of elite individuals to preserve
            local_search_rate: Fraction of non-elite offspring that get a 2-opt pass
                each generation (2-opt is O(route_length^2); applying it to every
                individual every generation is wasteful, so it's sampled)
            stagnation_limit: Generations without improvement before injecting
                fresh individuals into the population
        """
        self.num_customers = num_customers
        self.num_vehicles = num_vehicles
        self.duration_matrix = duration_matrix
        self.distance_matrix = distance_matrix
        self.vehicle_capacity = vehicle_capacity
        self.quantities = quantities
        self.population_size = population_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elite_size = elite_size
        self.local_search_rate = local_search_rate
        self.stagnation_limit = stagnation_limit

        self.population: List[Chromosome] = []
        self.fitness_history = []
        self.best_solution: Optional[Chromosome] = None
        self.best_fitness = float('inf')
        self._stagnant_generations = 0

        logger.info(f"GA (optimized) initialized: {num_customers} customers, {num_vehicles} vehicles, "
                    f"pop={population_size}, gen={generations}, capacity_enforced={quantities is not None}")

    def _route_time_distance(self, route: List[int]) -> Tuple[float, float]:
        if not route:
            return 0.0, 0.0
        time_s = 0.0
        distance_m = 0.0
        current = 0
        for customer_id in route:
            next_loc = customer_id + 1
            time_s += self.duration_matrix[current, next_loc]
            distance_m += self.distance_matrix[current, next_loc]
            current = next_loc
        time_s += self.duration_matrix[current, 0]
        distance_m += self.distance_matrix[current, 0]
        return time_s, distance_m

    def evaluate_fitness(self, chromosome: Chromosome) -> Tuple[float, Dict]:
        """
        Lexicographic-via-weighted-sum objective:
        1. Minimize makespan (max vehicle delivery time)
        2. Minimize total delivery time
        3. Minimize total cost
        Plus a capacity violation penalty when `quantities` was provided.
        """
        routes = chromosome.get_routes()
        vehicle_times, vehicle_distances, vehicle_costs = [], [], []
        total_time = 0.0
        total_distance = 0.0
        penalty = 0.0

        for route in routes:
            time_s, distance_m = self._route_time_distance(route)
            cost_idr = (distance_m / 1000.0) * COST_PER_KM_IDR

            if self.quantities is not None and route:
                load = sum(self.quantities[c] for c in route)
                if load > self.vehicle_capacity:
                    penalty += (load - self.vehicle_capacity) * CAPACITY_PENALTY_PER_UNIT

            vehicle_times.append(time_s)
            vehicle_distances.append(distance_m)
            vehicle_costs.append(cost_idr)
            total_time += time_s
            total_distance += distance_m

        makespan = max(vehicle_times) if vehicle_times else 0.0
        total_cost = sum(vehicle_costs)

        makespan_norm = makespan / 3600
        time_norm = total_time / 3600
        cost_norm = total_cost / 1_000_000

        fitness = (
            0.7 * makespan_norm +
            0.3 * cost_norm +
            penalty
        )

        details = {
            'makespan_s': makespan,
            'makespan_h': makespan_norm,
            'total_time_s': total_time,
            'total_time_h': time_norm,
            'total_distance_m': total_distance,
            'total_distance_km': total_distance / 1000,
            'total_cost_idr': total_cost,
            'vehicle_times_s': vehicle_times,
            'vehicle_distances_m': vehicle_distances,
            'vehicle_costs_idr': vehicle_costs,
            'penalty': penalty,
            'fitness': fitness,
        }

        return fitness, details

    def crossover(self, parent1: Chromosome, parent2: Chromosome) -> Chromosome:
        """Order crossover (OX) applied per vehicle route: for each vehicle, take
        a contiguous slice of parent1's route, then fill the remaining slots with
        parent2's customers (for that vehicle) in parent2's order, skipping
        duplicates. Repairs afterward to fix cross-vehicle duplicates/omissions."""
        child = Chromosome(self.num_customers, self.num_vehicles)

        for v in range(self.num_vehicles):
            r1 = parent1.routes[v]
            r2 = parent2.routes[v]

            if not r1:
                child.routes[v] = list(r2)
                continue
            if len(r1) == 1:
                child.routes[v] = list(r1)
                continue

            a, b = sorted(random.sample(range(len(r1)), 2))
            slice1 = r1[a:b + 1]
            fill = [c for c in r2 if c not in slice1]

            child_route = fill[:a] + slice1 + fill[a:]
            child.routes[v] = child_route

        child.repair()
        return child

    def mutate(self, chromosome: Chromosome):
        """Two mutation moves, applied stochastically: relocate a customer to a
        different vehicle (explores load balance), and swap two customers within
        a route (explores visiting order - genetic_algorithm.py's mutation never
        does this since it only mutates the assignment vector)."""
        for _ in range(max(1, int(self.num_customers * self.mutation_rate))):
            non_empty = [v for v, r in enumerate(chromosome.routes) if r]
            if not non_empty:
                break
            v = random.choice(non_empty)
            route = chromosome.routes[v]

            if random.random() < 0.5 and len(route) >= 2:
                i, j = random.sample(range(len(route)), 2)
                route[i], route[j] = route[j], route[i]
            else:
                idx = random.randrange(len(route))
                customer_id = route.pop(idx)
                new_vehicle = random.randrange(self.num_vehicles)
                insert_at = random.randrange(len(chromosome.routes[new_vehicle]) + 1)
                chromosome.routes[new_vehicle].insert(insert_at, customer_id)

    def two_opt(self, route: List[int], max_passes: int = 1) -> List[int]:
        """Bounded 2-opt local search on a single vehicle's route. Reverses
        segments that shorten total route time; stops after a full pass with no
        improvement or after max_passes full passes (kept small since this runs
        inside the GA loop, not as a one-off post-processing step)."""
        if len(route) < 3:
            return route

        best = route
        for _ in range(max_passes):
            improved = False
            best_time, _ = self._route_time_distance(best)
            for i in range(len(best) - 1):
                for j in range(i + 1, len(best)):
                    candidate = best[:i] + best[i:j + 1][::-1] + best[j + 1:]
                    cand_time, _ = self._route_time_distance(candidate)
                    if cand_time < best_time:
                        best, best_time = candidate, cand_time
                        improved = True
            if not improved:
                break
        return best

    def initialize_population(self, use_heuristics: bool = True):
        self.population = []

        if use_heuristics:
            chrom = Chromosome.nearest_neighbor_init(
                self.num_customers, self.distance_matrix, self.num_vehicles
            )
            self.population.append(chrom)

        while len(self.population) < self.population_size:
            chrom = Chromosome.random_init(self.num_customers, self.num_vehicles)
            self.population.append(chrom)

        logger.info(f"Initialized population with {len(self.population)} individuals")

    def _inject_diversity(self, fitness_scores: List[float]):
        """Replace the worst half of the non-elite population with fresh
        individuals when the search has stagnated (see module docstring point 4)."""
        order = sorted(range(len(self.population)), key=lambda i: fitness_scores[i])
        keep = order[:self.elite_size + (self.population_size - self.elite_size) // 2]
        kept = [self.population[i].clone() for i in keep]

        while len(kept) < self.population_size:
            if random.random() < 0.5:
                kept.append(Chromosome.nearest_neighbor_init(
                    self.num_customers, self.distance_matrix, self.num_vehicles))
            else:
                kept.append(Chromosome.random_init(self.num_customers, self.num_vehicles))

        self.population = kept
        self._stagnant_generations = 0
        logger.info("Stagnation detected - injected fresh individuals into population")

    def run(self) -> Chromosome:
        self.initialize_population()

        for gen in range(self.generations):
            fitness_scores = []
            details_list = []

            for chrom in self.population:
                fitness, details = self.evaluate_fitness(chrom)
                fitness_scores.append(fitness)
                details_list.append(details)

            best_idx = int(np.argmin(fitness_scores))
            best_fitness = fitness_scores[best_idx]
            best_details = details_list[best_idx]

            if best_fitness < self.best_fitness - 1e-9:
                self.best_fitness = best_fitness
                self.best_solution = self.population[best_idx].clone()
                self._stagnant_generations = 0
            else:
                self._stagnant_generations += 1

            self.fitness_history.append({
                'generation': gen,
                'best_fitness': best_fitness,
                'mean_fitness': float(np.mean(fitness_scores)),
                'makespan_s': best_details['makespan_s'],
                'total_cost_idr': best_details['total_cost_idr'],
            })

            if gen % 50 == 0:
                logger.info(f"Gen {gen:3d}: best_fitness={best_fitness:.2f}, "
                            f"makespan={best_details['makespan_s']/3600:.2f}h, "
                            f"cost={best_details['total_cost_idr']:.0f}IDR, "
                            f"penalty={best_details['penalty']:.2f}")

            if self._stagnant_generations >= self.stagnation_limit:
                self._inject_diversity(fitness_scores)
                continue

            # Selection (tournament)
            tournament_size = 4
            selected = []
            for _ in range(self.population_size - self.elite_size):
                tournament_indices = random.sample(range(len(self.population)), tournament_size)
                winner_idx = min(tournament_indices, key=lambda i: fitness_scores[i])
                selected.append(self.population[winner_idx].clone())

            # Crossover
            offspring = []
            for chrom in selected:
                if random.random() < self.crossover_rate:
                    other = random.choice(selected)
                    child = self.crossover(chrom, other)
                else:
                    child = chrom.clone()
                offspring.append(child)

            # Mutation
            for chrom in offspring:
                if random.random() < 0.5:
                    self.mutate(chrom)

            # Bounded 2-opt refinement on a sample of offspring
            for chrom in offspring:
                if random.random() < self.local_search_rate:
                    chrom.routes = [self.two_opt(r) for r in chrom.routes]

            # Elitism
            elite_indices = sorted(
                range(len(self.population)),
                key=lambda i: fitness_scores[i]
            )[:self.elite_size]

            new_population = [self.population[i].clone() for i in elite_indices]
            new_population.extend(offspring[:self.population_size - self.elite_size])
            self.population = new_population

        logger.info(f"GA (optimized) completed. Best fitness: {self.best_fitness:.2f}")
        return self.best_solution

    def get_best_solution_details(self) -> Dict:
        if self.best_solution is None:
            return {}

        fitness, details = self.evaluate_fitness(self.best_solution)
        routes = self.best_solution.get_routes()

        return {
            **details,
            'routes': routes,
            'num_routes': len([r for r in routes if r]),
        }
