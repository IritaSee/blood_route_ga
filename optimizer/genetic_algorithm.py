"""
Genetic Algorithm for 2-vehicle blood supply routing optimization.
Minimizes delivery time (primary) and cost (secondary).
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import random

logger = logging.getLogger(__name__)

# Constants
FUEL_PRICE_IDR_PER_LITER = 12750
FUEL_EFFICIENCY_KM_PER_LITER = 9
COST_PER_KM_IDR = FUEL_PRICE_IDR_PER_LITER / FUEL_EFFICIENCY_KM_PER_LITER  # ≈ 1416.67


@dataclass
class VehicleRoute:
    """Represents a single vehicle's route."""
    vehicle_id: int
    facility_sequence: List[int]  # Indices of facilities in visit order
    quantities: List[float]  # Quantities delivered at each stop
    total_distance_m: float
    total_duration_s: float
    total_cost_idr: float
    load: float  # Current load
    
    def summary(self) -> Dict:
        """Return route summary."""
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
    GA chromosome for 2-vehicle routing.
    Encoding: Assignment vector + per-vehicle ordered sequences
    """
    
    def __init__(self, num_customers: int, num_vehicles: int = 2):
        """
        Initialize chromosome.
        
        Args:
            num_customers: Number of delivery locations (excluding depot)
            num_vehicles: Number of vehicles (default: 2)
        """
        self.num_customers = num_customers
        self.num_vehicles = num_vehicles
        self.vehicle_assignments = np.zeros(num_customers, dtype=int)  # Which vehicle for each customer
        self.routes = [[] for _ in range(num_vehicles)]
        
    @classmethod
    def random_init(cls, num_customers: int, num_vehicles: int = 2) -> 'Chromosome':
        """Create random chromosome."""
        chrom = cls(num_customers, num_vehicles)
        # Randomly assign customers to vehicles
        chrom.vehicle_assignments = np.random.randint(0, num_vehicles, num_customers)
        # Build routes from assignments
        chrom._build_routes_from_assignments()
        return chrom
    
    @classmethod
    def nearest_neighbor_init(cls, num_customers: int, distance_matrix: np.ndarray, 
                             num_vehicles: int = 2) -> 'Chromosome':
        """Initialize using nearest-neighbor heuristic."""
        chrom = cls(num_customers, num_vehicles)
        
        # Start from depot (index 0), but we optimize customers only
        # Simple partition: alternate assignment to vehicles
        unassigned = set(range(num_customers))
        for vehicle_id in range(num_vehicles):
            route = []
            current = 0  # Depot
            
            while unassigned:
                # Find nearest unassigned customer from current
                nearest = min(unassigned, key=lambda x: distance_matrix[current, x + 1])
                route.append(nearest)
                chrom.vehicle_assignments[nearest] = vehicle_id
                unassigned.remove(nearest)
                current = nearest + 1
            
            chrom.routes[vehicle_id] = route
        
        return chrom
    
    def _build_routes_from_assignments(self):
        """Build route lists from assignment vector."""
        self.routes = [[] for _ in range(self.num_vehicles)]
        for customer_id, vehicle_id in enumerate(self.vehicle_assignments):
            self.routes[vehicle_id].append(customer_id)
    
    def get_routes(self) -> List[List[int]]:
        """Get list of routes (customer sequences per vehicle)."""
        return self.routes
    
    def set_routes(self, routes: List[List[int]]):
        """Set routes and update assignments."""
        self.routes = routes
        self.vehicle_assignments = np.zeros(self.num_customers, dtype=int)
        for vehicle_id, route in enumerate(routes):
            for customer_id in route:
                self.vehicle_assignments[customer_id] = vehicle_id
    
    def clone(self) -> 'Chromosome':
        """Create a copy of this chromosome."""
        chrom = Chromosome(self.num_customers, self.num_vehicles)
        chrom.vehicle_assignments = self.vehicle_assignments.copy()
        chrom.routes = [route.copy() for route in self.routes]
        return chrom


class GeneticAlgorithm:
    """
    GA for 2-vehicle routing optimization.
    Objectives: minimize delivery time (primary), then cost (secondary).
    """
    
    def __init__(self, num_customers: int, num_vehicles: int, 
                 duration_matrix: np.ndarray, distance_matrix: np.ndarray,
                 vehicle_capacity: float = 100.0,
                 population_size: int = 150, 
                 generations: int = 800,
                 crossover_rate: float = 0.8,
                 mutation_rate: float = 0.1,
                 elite_size: int = 15):
        """
        Initialize GA.
        
        Args:
            num_customers: Number of delivery locations
            num_vehicles: Number of vehicles
            duration_matrix: (n_locations x n_locations) duration in seconds
            distance_matrix: (n_locations x n_locations) distance in meters
            vehicle_capacity: Capacity per vehicle (units)
            population_size: Population size
            generations: Number of generations
            crossover_rate: Crossover probability
            mutation_rate: Mutation probability per gene
            elite_size: Number of elite individuals to preserve
        """
        self.num_customers = num_customers
        self.num_vehicles = num_vehicles
        self.duration_matrix = duration_matrix
        self.distance_matrix = distance_matrix
        self.vehicle_capacity = vehicle_capacity
        self.population_size = population_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elite_size = elite_size
        
        self.population: List[Chromosome] = []
        self.fitness_history = []
        self.best_solution: Optional[Chromosome] = None
        self.best_fitness = float('inf')
        
        logger.info(f"GA initialized: {num_customers} customers, {num_vehicles} vehicles, "
                   f"pop={population_size}, gen={generations}")
    
    def evaluate_fitness(self, chromosome: Chromosome) -> Tuple[float, Dict]:
        """
        Evaluate fitness of a chromosome.
        
        Lexicographic objective:
        1. Minimize makespan (max vehicle delivery time)
        2. Minimize total delivery time
        3. Minimize total cost
        
        Returns:
            Tuple of (fitness_scalar, details_dict)
        """
        routes = chromosome.get_routes()
        vehicle_times = []
        vehicle_distances = []
        vehicle_costs = []
        total_time = 0
        total_distance = 0
        penalty = 0
        
        for vehicle_id, route in enumerate(routes):
            if not route:
                vehicle_times.append(0)
                vehicle_distances.append(0)
                vehicle_costs.append(0)
                continue
            
            # Calculate time and distance for this route
            # Route: depot -> customers -> depot
            time_s = 0
            distance_m = 0
            
            # Depot (index 0) to first customer
            current = 0
            for customer_id in route:
                next_loc = customer_id + 1  # Customers are 1-indexed in matrix
                time_s += self.duration_matrix[current, next_loc]
                distance_m += self.distance_matrix[current, next_loc]
                current = next_loc
            
            # Last customer back to depot
            time_s += self.duration_matrix[current, 0]
            distance_m += self.distance_matrix[current, 0]
            
            cost_idr = (distance_m / 1000.0) * COST_PER_KM_IDR
            
            vehicle_times.append(time_s)
            vehicle_distances.append(distance_m)
            vehicle_costs.append(cost_idr)
            total_time += time_s
            total_distance += distance_m
        
        # Makespan (max time among vehicles)
        makespan = max(vehicle_times) if vehicle_times else 0
        
        # Total cost
        total_cost = sum(vehicle_costs)
        
        # Weighted sum fitness (lower is better): Time 70%, Cost 30%
        # Scale for numerical stability
        makespan_norm = makespan / 3600  # Convert to hours
        time_norm = total_time / 3600
        cost_norm = total_cost / 1000000  # Convert to millions of IDR
        
        # Fitness: 70% Time + 30% Cost (per new specifications)
        fitness = (
            0.7 * makespan_norm +  # 70% weight on time (makespan)
            0.3 * cost_norm        # 30% weight on cost
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
        """
        Order crossover (OX) between two parents.
        """
        child = parent1.clone()
        
        # For simplicity, do uniform assignment crossover
        for i in range(self.num_customers):
            if random.random() < 0.5:
                child.vehicle_assignments[i] = parent2.vehicle_assignments[i]
        
        child._build_routes_from_assignments()
        return child
    
    def mutate(self, chromosome: Chromosome):
        """
        Mutation: move a random customer to a different vehicle.
        """
        for _ in range(max(1, int(self.num_customers * self.mutation_rate))):
            # Pick random customer and random vehicle
            customer_id = random.randint(0, self.num_customers - 1)
            new_vehicle = random.randint(0, self.num_vehicles - 1)
            
            # Move to new vehicle
            chromosome.vehicle_assignments[customer_id] = new_vehicle
        
        chromosome._build_routes_from_assignments()
    
    def initialize_population(self, use_heuristics: bool = True):
        """Initialize population with mix of random and heuristic solutions."""
        self.population = []
        
        # Add heuristic solution if available
        if use_heuristics:
            chrom = Chromosome.nearest_neighbor_init(
                self.num_customers, self.distance_matrix, self.num_vehicles
            )
            self.population.append(chrom)
        
        # Fill rest with random
        while len(self.population) < self.population_size:
            chrom = Chromosome.random_init(self.num_customers, self.num_vehicles)
            self.population.append(chrom)
        
        logger.info(f"Initialized population with {len(self.population)} individuals")
    
    def run(self) -> Chromosome:
        """Execute GA optimization."""
        self.initialize_population()
        
        for gen in range(self.generations):
            # Evaluate fitness
            fitness_scores = []
            details_list = []
            
            for chrom in self.population:
                fitness, details = self.evaluate_fitness(chrom)
                fitness_scores.append(fitness)
                details_list.append(details)
            
            # Track best
            best_idx = np.argmin(fitness_scores)
            best_fitness = fitness_scores[best_idx]
            best_details = details_list[best_idx]
            
            if best_fitness < self.best_fitness:
                self.best_fitness = best_fitness
                self.best_solution = self.population[best_idx].clone()
            
            self.fitness_history.append({
                'generation': gen,
                'best_fitness': best_fitness,
                'mean_fitness': np.mean(fitness_scores),
                'makespan_s': best_details['makespan_s'],
                'total_cost_idr': best_details['total_cost_idr'],
            })
            
            if gen % 50 == 0:
                logger.info(f"Gen {gen:3d}: best_fitness={best_fitness:.2f}, "
                          f"makespan={best_details['makespan_s']/3600:.2f}h, "
                          f"cost={best_details['total_cost_idr']:.0f}IDR")
            
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
            
            # Elitism: preserve best solutions
            elite_indices = sorted(
                range(len(self.population)), 
                key=lambda i: fitness_scores[i]
            )[:self.elite_size]
            
            new_population = [self.population[i].clone() for i in elite_indices]
            new_population.extend(offspring[:self.population_size - self.elite_size])
            self.population = new_population
        
        logger.info(f"GA completed. Best fitness: {self.best_fitness:.2f}")
        return self.best_solution
    
    def get_best_solution_details(self) -> Dict:
        """Get details of best solution found."""
        if self.best_solution is None:
            return {}
        
        fitness, details = self.evaluate_fitness(self.best_solution)
        routes = self.best_solution.get_routes()
        
        return {
            **details,
            'routes': routes,
            'num_routes': len([r for r in routes if r]),
        }
