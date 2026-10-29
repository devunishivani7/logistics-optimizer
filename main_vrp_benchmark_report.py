import time
import csv
from pathlib import Path
import numpy as np

print("Indigenous Hybrid CPU-GPU Optimization Solver")
print("Domain: Logistics")
print("Problem: Vehicle Routing Problem (VRP)")

# Warehouse and customer coordinates: (x, y)
locations = np.array([
    [0, 0],   # Warehouse
    [2, 3],   # Customer 1
    [5, 4],   # Customer 2
    [1, 6],   # Customer 3
    [7, 2]    # Customer 4
])

print("\nWarehouse and customer coordinates:")
print(locations)

print("\nNumber of customers:", len(locations) - 1)
print("CPU environment is ready!")

# Calculate the distance matrix
difference = locations[:, np.newaxis, :] - locations[np.newaxis, :, :]

distance_matrix = np.sqrt(np.sum(difference ** 2, axis=2))

print("\nDistance Matrix:")
print(np.round(distance_matrix, 2))

# Customer demands
# Index 0 represents the warehouse
demands = np.array([0, 2, 3, 2, 4])

# Vehicle capacity
vehicle_capacity = 6

print("\nCustomer Demands:")
for i in range(1, len(locations)):
    print(f"Customer {i}: {demands[i]} units")

print("\nVehicle capacity:", vehicle_capacity)
print("Total customer demand:", np.sum(demands))
print("Minimum vehicles required (capacity only):",
      int(np.ceil(np.sum(demands) / vehicle_capacity)))

# Sample delivery route
route = [0, 1, 2, 3, 4, 0]

# Calculate total distance of the route
total_distance = 0

for i in range(len(route) - 1):
    current_location = route[i]
    next_location = route[i + 1]

    total_distance += distance_matrix[
        current_location, next_location
    ]

print("\nSample Route:", " -> ".join(map(str, route)))
print("Total route distance:", round(total_distance, 2))

# Check whether a route respects vehicle capacity
route_customers = route[1:-1]

route_demand = sum(demands[customer] for customer in route_customers)

print("\nRoute demand:", route_demand)
print("Vehicle capacity:", vehicle_capacity)

if route_demand <= vehicle_capacity:
    print("Route is FEASIBLE: capacity constraint satisfied.")
else:
    print("Route is INFEASIBLE: vehicle capacity exceeded.")

# Build capacity-feasible routes using a greedy approach
routes = []
current_route = [0]
current_load = 0

for customer in range(1, len(locations)):
    demand = demands[customer]

    if current_load + demand > vehicle_capacity:
        current_route.append(0)
        routes.append(current_route)

        current_route = [0]
        current_load = 0

    current_route.append(customer)
    current_load += demand

current_route.append(0)
routes.append(current_route)

print("\nCapacity-Feasible Routes:")

for vehicle_id, vehicle_route in enumerate(routes, start=1):
    load = sum(demands[c] for c in vehicle_route)
    print(f"Vehicle {vehicle_id}: {vehicle_route}, Load: {load}/{vehicle_capacity}")

# Calculate total distance for all vehicle routes
total_distance_all_vehicles = 0

print("\nRoute Distance Analysis:")

for vehicle_id, vehicle_route in enumerate(routes, start=1):
    route_distance = 0

    for i in range(len(vehicle_route) - 1):
        current_location = vehicle_route[i]
        next_location = vehicle_route[i + 1]

        route_distance += distance_matrix[
            current_location, next_location
        ]

    total_distance_all_vehicles += route_distance

    print(
        f"Vehicle {vehicle_id} distance: "
        f"{route_distance:.2f}"
    )

print(
    "\nTotal fleet distance:",
    round(total_distance_all_vehicles, 2)
)
print("Number of vehicles used:", len(routes))

# Validate the complete solution
visited_customers = []

for vehicle_route in routes:
    # Exclude the starting and ending warehouse
    customers = vehicle_route[1:-1]
    visited_customers.extend(customers)

# Check that every customer is visited exactly once
all_customers = list(range(1, len(locations)))

customers_visited_once = (
    sorted(visited_customers) == all_customers
)

# Check that every vehicle respects its capacity
capacity_valid = all(
    sum(demands[c] for c in vehicle_route[1:-1])
    <= vehicle_capacity
    for vehicle_route in routes
)

print("\nSolution Validation:")
print("Every customer visited exactly once:", customers_visited_once)
print("All vehicle capacities respected:", capacity_valid)

if customers_visited_once and capacity_valid:
    print("Solution status: VALID")
else:
    print("Solution status: INVALID")

def calculate_route_distance(route, distance_matrix):
    """Calculate the total distance of a route."""
    return sum(
        distance_matrix[route[i], route[i + 1]]
        for i in range(len(route) - 1)
    )


def two_opt(route, distance_matrix):
    """Improve a route by reversing customer subsequences."""
    best_route = route.copy()
    best_distance = calculate_route_distance(
        best_route, distance_matrix
    )

    improved = True

    while improved:
        improved = False

        # Keep the first and last warehouse fixed
        for i in range(1, len(best_route) - 2):
            for j in range(i + 1, len(best_route) - 1):
                candidate = (
                    best_route[:i]
                    + best_route[i:j + 1][::-1]
                    + best_route[j + 1:]
                )

                candidate_distance = calculate_route_distance(
                    candidate, distance_matrix
                )

                if candidate_distance < best_distance - 1e-9:
                    best_route = candidate
                    best_distance = candidate_distance
                    improved = True
                    break

            if improved:
                break

    return best_route

# Improve each vehicle's route using 2-opt
optimized_routes = []

for vehicle_route in routes:
    improved_route = two_opt(vehicle_route, distance_matrix)
    optimized_routes.append(improved_route)

print("\n2-opt Optimization Results:")

optimized_total_distance = 0

for vehicle_id, vehicle_route in enumerate(
    optimized_routes, start=1
):
    distance = calculate_route_distance(
        vehicle_route, distance_matrix
    )
    optimized_total_distance += distance

    print(f"Vehicle {vehicle_id}: {vehicle_route}")
    print(f"Distance: {distance:.2f}")

print(
    "\nOptimized fleet distance:",
    round(optimized_total_distance, 2)
)

# Compare baseline and optimized solutions
improvement = total_distance_all_vehicles - optimized_total_distance

if total_distance_all_vehicles > 0:
    improvement_percent = (
        improvement / total_distance_all_vehicles
    ) * 100
else:
    improvement_percent = 0

print("\nOptimization Summary:")
print("Baseline distance:", round(total_distance_all_vehicles, 2))
print("Optimized distance:", round(optimized_total_distance, 2))
print("Distance saved:", round(improvement, 2))
print("Improvement percentage:", round(improvement_percent, 2), "%")


def decode_chromosome(chromosome, demands, vehicle_capacity):
    """Convert a customer ordering into capacity-feasible routes."""
    routes = []
    current_route = [0]
    current_load = 0

    for customer in chromosome:
        demand = demands[customer]

        if demand > vehicle_capacity:
            raise ValueError(
                f"Customer {customer} exceeds vehicle capacity."
            )

        if current_load + demand > vehicle_capacity:
            current_route.append(0)
            routes.append(current_route)

            current_route = [0]
            current_load = 0

        current_route.append(customer)
        current_load += demand

    current_route.append(0)
    routes.append(current_route)

    return routes


# Test the chromosome decoder
test_chromosome = [1, 2, 3, 4]

decoded_routes = decode_chromosome(
    test_chromosome, demands, vehicle_capacity
)

print("\nChromosome Decoder Test:")
print("Chromosome:", test_chromosome)
print("Decoded routes:", decoded_routes)

def calculate_fitness(
    chromosome, demands, vehicle_capacity, distance_matrix
):
    """Return total fleet distance for a chromosome."""
    routes = decode_chromosome(
        chromosome, demands, vehicle_capacity
    )

    total_distance = 0

    for route in routes:
        total_distance += calculate_route_distance(
            route, distance_matrix
        )

    return total_distance


# Test the fitness function
test_chromosome = [1, 2, 3, 4]

test_cost = calculate_fitness(
    test_chromosome,
    demands,
    vehicle_capacity,
    distance_matrix
)

print("\nFitness Function Test:")
print("Chromosome:", test_chromosome)
print("Total distance (cost):", round(test_cost, 2))

import random

def create_initial_population(num_customers, population_size):
    """Create random customer orderings."""
    population = []

    customers = list(range(1, num_customers + 1))

    for _ in range(population_size):
        chromosome = customers.copy()
        random.shuffle(chromosome)
        population.append(chromosome)

    return population


# Generate the initial population
population_size = 10

population = create_initial_population(
    len(locations) - 1,
    population_size
)

print("\nInitial Genetic Algorithm Population:")

for index, chromosome in enumerate(population, start=1):
    print(f"Chromosome {index}: {chromosome}")

# Evaluate every chromosome in the population
population_costs = []

for chromosome in population:
    cost = calculate_fitness(
        chromosome,
        demands,
        vehicle_capacity,
        distance_matrix
    )
    population_costs.append(cost)

print("\nPopulation Fitness Evaluation:")

for index, (chromosome, cost) in enumerate(
    zip(population, population_costs), start=1
):
    print(f"Chromosome {index}: {chromosome}, Cost: {cost:.2f}")

# Find the chromosome with the minimum cost
best_index = population_costs.index(min(population_costs))
best_chromosome = population[best_index]
best_cost = population_costs[best_index]

print("\nBest Candidate:")
print("Chromosome:", best_chromosome)
print("Total distance:", round(best_cost, 2))

def tournament_selection(population, population_costs, tournament_size=3):
    """Select the lowest-cost candidate from a random tournament."""
    indices = random.sample(
        range(len(population)),
        min(tournament_size, len(population))
    )

    winner_index = min(
        indices,
        key=lambda index: population_costs[index]
    )

    return population[winner_index].copy()


# Test tournament selection
selected_parent = tournament_selection(
    population, population_costs
)

print("\nTournament Selection Test:")
print("Selected parent:", selected_parent)
print(
    "Selected parent cost:",
    round(
        calculate_fitness(
            selected_parent,
            demands,
            vehicle_capacity,
            distance_matrix
        ),
        2
    )
)

def order_crossover(parent1, parent2):
    """Create a valid child using order crossover (OX)."""
    size = len(parent1)

    if size < 2:
        return parent1.copy()

    start, end = sorted(random.sample(range(size), 2))

    child = [None] * size

    # Copy a segment from parent 1
    child[start:end + 1] = parent1[start:end + 1]

    # Fill remaining positions in parent 2's order
    remaining = [
        customer
        for customer in parent2
        if customer not in child
    ]

    fill_positions = [
        i for i in range(size)
        if child[i] is None
    ]

    for position, customer in zip(fill_positions, remaining):
        child[position] = customer

    return child


# Test crossover
parent1 = [1, 2, 3, 4]
parent2 = [3, 4, 1, 2]

child = order_crossover(parent1, parent2)

print("\nCrossover Test:")
print("Parent 1:", parent1)
print("Parent 2:", parent2)
print("Child:", child)
print("Child contains every customer once:", sorted(child) == [1, 2, 3, 4])

def mutate_chromosome(chromosome, mutation_rate=0.2):
    """Randomly swap two customers with a given probability."""
    mutated = chromosome.copy()

    if len(mutated) >= 2 and random.random() < mutation_rate:
        i, j = random.sample(range(len(mutated)), 2)
        mutated[i], mutated[j] = mutated[j], mutated[i]

    return mutated


# Test mutation
test_chromosome = [1, 2, 3, 4]

mutated_chromosome = mutate_chromosome(
    test_chromosome,
    mutation_rate=1.0
)

print("\nMutation Test:")
print("Original chromosome:", test_chromosome)
print("Mutated chromosome:", mutated_chromosome)
print(
    "All customers preserved:",
    sorted(mutated_chromosome) == [1, 2, 3, 4]
)

def create_gpu_fitness_evaluator(
    demands,
    vehicle_capacity,
    distance_matrix,
    max_population_size=100
):
    """Create an OpenCL evaluator with reusable buffers for population batches."""
    import pyopencl as cl

    distance_matrix = np.ascontiguousarray(distance_matrix, dtype=np.float32)
    flat_distances = distance_matrix.reshape(-1)
    n = len(distance_matrix)
    num_customers = len(demands) - 1
    # A decoded route list has at most one [depot, customer, depot] per customer.
    max_stops = max(3 * num_customers, 3)

    gpu_devices = []
    for platform in cl.get_platforms():
        gpu_devices.extend(platform.get_devices(device_type=cl.device_type.GPU))
    if not gpu_devices:
        raise RuntimeError("No OpenCL GPU available.")

    device = gpu_devices[0]
    context = cl.Context([device])
    queue = cl.CommandQueue(context)
    mf = cl.mem_flags

    kernel_code = """
    __kernel void evaluate_routes(
        __global const int *route_data,
        __global const int *route_lengths,
        __global const float *distances,
        __global float *fitness,
        const int n,
        const int max_stops
    ) {
        int c = get_global_id(0);
        float total = 0.0f;
        for (int i = 0; i < route_lengths[c] - 1; i++) {
            int from = route_data[c * max_stops + i];
            int to = route_data[c * max_stops + i + 1];
            total += distances[from * n + to];
        }
        fitness[c] = total;
    }
    """
    program = cl.Program(context, kernel_code).build()
    kernel = cl.Kernel(program, "evaluate_routes")

    # Fixed distance matrix is uploaded once; batch buffers are reused.
    distance_buffer = cl.Buffer(
        context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=flat_distances
    )
    route_data_host = np.zeros((max_population_size, max_stops), dtype=np.int32)
    route_lengths_host = np.zeros(max_population_size, dtype=np.int32)
    results_host = np.empty(max_population_size, dtype=np.float32)

    route_buffer = cl.Buffer(context, mf.READ_ONLY, route_data_host.nbytes)
    length_buffer = cl.Buffer(context, mf.READ_ONLY, route_lengths_host.nbytes)
    fitness_buffer = cl.Buffer(context, mf.WRITE_ONLY, results_host.nbytes)

    def evaluate_population(population):
        if not population:
            return []
        batch_size = len(population)
        if batch_size > max_population_size:
            raise ValueError(
                f"Population size {batch_size} exceeds evaluator capacity "
                f"{max_population_size}."
            )

        route_data_host[:batch_size, :] = 0
        route_lengths_host[:batch_size] = 0

        for i, chromosome in enumerate(population):
            decoded = decode_chromosome(
                chromosome, demands, vehicle_capacity
            )
            route = [node for one_route in decoded for node in one_route]
            if len(route) > max_stops:
                raise ValueError(
                    f"Decoded route length {len(route)} exceeds {max_stops}."
                )
            route_data_host[i, :len(route)] = route
            route_lengths_host[i] = len(route)

        cl.enqueue_copy(
            queue, route_buffer, route_data_host[:batch_size], is_blocking=False
        )
        cl.enqueue_copy(
            queue, length_buffer, route_lengths_host[:batch_size], is_blocking=False
        )
        kernel(
            queue, (batch_size,), None,
            route_buffer, length_buffer, distance_buffer, fitness_buffer,
            np.int32(n), np.int32(max_stops)
        )
        cl.enqueue_copy(
            queue, results_host[:batch_size], fitness_buffer, is_blocking=False
        ).wait()
        return results_host[:batch_size].astype(np.float64).tolist()

    print("GPU fitness evaluator (reusable buffers):", device.name)
    return evaluate_population


def genetic_algorithm(
    demands,
    vehicle_capacity,
    distance_matrix,
    population_size=20,
    generations=100,
    mutation_rate=0.2,
    use_gpu=False,
    gpu_evaluator=None
):
    num_customers = len(demands) - 1

    population = create_initial_population(
        num_customers,
        population_size
    )

    # Initialize the GPU evaluator once per GA run.
    if use_gpu and gpu_evaluator is None:
        gpu_evaluator = create_gpu_fitness_evaluator(
            demands,
            vehicle_capacity,
            distance_matrix,
            max_population_size=population_size
        )

    best_chromosome = None
    best_cost = float("inf")
    history = []

    for generation in range(generations):
        # Evaluate the whole population on CPU or GPU.
        if use_gpu:
            population_costs = gpu_evaluator(population)
        else:
            population_costs = [
                calculate_fitness(
                    chromosome,
                    demands,
                    vehicle_capacity,
                    distance_matrix
                )
                for chromosome in population
            ]

        best_index = population_costs.index(min(population_costs))
        generation_best = population[best_index]
        generation_cost = population_costs[best_index]

        if generation_cost < best_cost:
            best_cost = generation_cost
            best_chromosome = generation_best.copy()

        history.append(best_cost)

        # Elitism: preserve the best chromosome found so far.
        new_population = [best_chromosome.copy()]

        while len(new_population) < population_size:
            parent1 = tournament_selection(
                population, population_costs
            )
            parent2 = tournament_selection(
                population, population_costs
            )

            child = order_crossover(parent1, parent2)
            child = mutate_chromosome(child, mutation_rate)
            new_population.append(child)

        population = new_population

    return best_chromosome, best_cost, history

# Run the complete genetic algorithm
ga_chromosome, ga_cost, history = genetic_algorithm(
    demands,
    vehicle_capacity,
    distance_matrix
)

print("\nGenetic Algorithm Results:")
print("Best customer order:", ga_chromosome)

ga_routes = decode_chromosome(
    ga_chromosome,
    demands,
    vehicle_capacity
)

print("Best delivery routes:", ga_routes)
print("Total fleet distance:", round(ga_cost, 2))
# Compare baseline and Genetic Algorithm
baseline_distance = sum(
    calculate_route_distance(route, distance_matrix)
    for route in routes
)

improvement = baseline_distance - ga_cost

improvement_percentage = (
    (improvement / baseline_distance) * 100
)

print("\nPerformance Comparison:")
print("Baseline distance:", round(baseline_distance, 2))
print("GA distance:", round(ga_cost, 2))
print("Distance saved:", round(improvement, 2))
print("Improvement:", round(improvement_percentage, 2), "%")
print("\nGA Convergence:")
print("First generation best:", round(history[0], 2))
print("Final generation best:", round(history[-1], 2))
print("Number of generations:", len(history))
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 5))

plt.plot(
    range(1, len(history) + 1),
    history,
    marker="o",
    markersize=3,
    label="Best distance"
)

plt.xlabel("Generation")
plt.ylabel("Best total fleet distance")
plt.title("Genetic Algorithm Convergence")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("ga_convergence.png", dpi=300)
plt.show()

# Larger test dataset: 20 customers + 1 warehouse
np.random.seed(42)

large_locations = np.vstack((
    np.array([[50, 50]]),
    np.random.randint(0, 101, size=(20, 2))
))

large_demands = np.concatenate((
    np.array([0]),
    np.random.randint(1, 5, size=20)
))

large_vehicle_capacity = 10

large_distance_matrix = np.zeros((21, 21))

for i in range(21):
    for j in range(21):
        large_distance_matrix[i, j] = np.linalg.norm(
            large_locations[i] - large_locations[j]
        )

print("\nLarger Dataset:")
print("Number of customers:", len(large_locations) - 1)
print("Number of vehicles needed at minimum:", 
      int(np.ceil(large_demands[1:].sum() / large_vehicle_capacity)))
print("Total customer demand:", large_demands[1:].sum())
# Run GA on the larger dataset
large_ga_chromosome, large_ga_cost, large_history = genetic_algorithm(
    demands=large_demands,
    vehicle_capacity=large_vehicle_capacity,
    distance_matrix=large_distance_matrix,
    population_size=1000,
    generations=10,
    mutation_rate=0.2,
    use_gpu=True
)

large_ga_routes = decode_chromosome(
    large_ga_chromosome,
    large_demands,
    large_vehicle_capacity
)

print("\n20-Customer GA Results:")
print("Best customer order:", large_ga_chromosome)
print("Number of vehicles used:", len(large_ga_routes))
print("Total fleet distance:", round(large_ga_cost, 2))
print("First generation best:", round(large_history[0], 2))
print("Final generation best:", round(large_history[-1], 2))
# Greedy baseline for the 20-customer dataset
large_greedy_routes = []
current_route = [0]
current_load = 0

for customer in range(1, len(large_locations)):
    demand = large_demands[customer]

    if current_load + demand > large_vehicle_capacity:
        current_route.append(0)
        large_greedy_routes.append(current_route)

        current_route = [0]
        current_load = 0

    current_route.append(customer)
    current_load += demand

current_route.append(0)
large_greedy_routes.append(current_route)

# Calculate baseline distance
large_greedy_distance = sum(
    calculate_route_distance(route, large_distance_matrix)
    for route in large_greedy_routes
)

# Compare the two methods
large_improvement = (
    (large_greedy_distance - large_ga_cost)
    / large_greedy_distance
) * 100

print("\n20-Customer Performance Comparison:")
print("Greedy vehicles used:", len(large_greedy_routes))
print("Greedy distance:", round(large_greedy_distance, 2))
print("GA vehicles used:", len(large_ga_routes))
print("GA distance:", round(large_ga_cost, 2))
print("Distance improvement:", round(large_improvement, 2), "%")

# Visualize the 20-customer GA solution
plt.figure(figsize=(10, 8))

# Plot warehouse
plt.scatter(
    large_locations[0, 0],
    large_locations[0, 1],
    color="red",
    marker="s",
    s=150,
    label="Warehouse",
    zorder=5
)

# Plot customers
plt.scatter(
    large_locations[1:, 0],
    large_locations[1:, 1],
    color="blue",
    marker="o",
    s=60,
    label="Customers"
)

# Label each location
for i, (x, y) in enumerate(large_locations):
    plt.annotate(
        str(i),
        (x, y),
        xytext=(5, 5),
        textcoords="offset points"
    )

# Draw each vehicle route
for vehicle_number, route in enumerate(large_ga_routes, start=1):
    route_coordinates = large_locations[route]

    plt.plot(
        route_coordinates[:, 0],
        route_coordinates[:, 1],
        marker=".",
        linewidth=1.5,
        label=f"Vehicle {vehicle_number}"
    )

plt.title("20-Customer Vehicle Routing Solution")
plt.xlabel("X Coordinate")
plt.ylabel("Y Coordinate")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("ga_routes_20_customers.png", dpi=300)
plt.show()
# Validate the larger GA solution
visited_customers = []

capacity_valid = True

for route in large_ga_routes:
    route_load = sum(large_demands[c] for c in route if c != 0)
    visited_customers.extend(c for c in route if c != 0)

    if route_load > large_vehicle_capacity:
        capacity_valid = False

all_customers_served = (
    sorted(visited_customers)
    == list(range(1, len(large_locations)))
)

print("\n20-Customer Solution Validation:")
print("All customers served exactly once:", all_customers_served)
print("Vehicle capacities respected:", capacity_valid)
print("Total customers visited:", len(visited_customers))
print("Total vehicles used:", len(large_ga_routes))
print("All constraints satisfied:", all_customers_served and capacity_valid)
# Benchmark the Genetic Algorithm
start_time = time.perf_counter()

benchmark_chromosome, benchmark_cost, benchmark_history = genetic_algorithm(
    demands=large_demands,
    vehicle_capacity=large_vehicle_capacity,
    distance_matrix=large_distance_matrix,
    population_size=100,
    generations=300,
    mutation_rate=0.2
)

end_time = time.perf_counter()

execution_time = end_time - start_time

print("\nGA Performance Benchmark:")
print("Customers:", len(large_locations) - 1)
print("Population size:", 100)
print("Generations:", 300)
print("Best distance:", round(benchmark_cost, 2))
print("Execution time:", round(execution_time, 4), "seconds")
# Repeat the GA benchmark 5 times
benchmark_times = []
benchmark_distances = []

for trial in range(5):
    start_time = time.perf_counter()

    _, trial_cost, _ = genetic_algorithm(
        demands=large_demands,
        vehicle_capacity=large_vehicle_capacity,
        distance_matrix=large_distance_matrix,
        population_size=100,
        generations=300,
        mutation_rate=0.2
    )

    elapsed = time.perf_counter() - start_time

    benchmark_times.append(elapsed)
    benchmark_distances.append(trial_cost)

    print(
        f"Trial {trial + 1}: "
        f"Distance = {trial_cost:.2f}, "
        f"Time = {elapsed:.4f} seconds"
    )

print("\nRepeated Benchmark Summary:")
print("Average distance:", round(np.mean(benchmark_distances), 2))
print("Best distance:", round(min(benchmark_distances), 2))
print("Worst distance:", round(max(benchmark_distances), 2))
print("Average execution time:", round(np.mean(benchmark_times), 4), "seconds")


# ------------------------------------------------------------
# FAIR CPU vs GPU GENETIC ALGORITHM BENCHMARK
# Same dataset, population size, generations, mutation rate,
# and paired random seeds for CPU/GPU runs.
# ------------------------------------------------------------
COMPARISON_TRIALS = 5
COMPARISON_POPULATION = 100
COMPARISON_GENERATIONS = 300
COMPARISON_MUTATION_RATE = 0.2
COMPARISON_SEEDS = [101, 202, 303, 404, 505]

cpu_times = []
gpu_times = []
cpu_distances = []
gpu_distances = []

print("\n" + "=" * 60)
print("FAIR CPU vs GPU GENETIC ALGORITHM BENCHMARK")
print("=" * 60)
print("Customers:", len(large_locations) - 1)
print("Population size:", COMPARISON_POPULATION)
print("Generations:", COMPARISON_GENERATIONS)
print("Trials:", COMPARISON_TRIALS)
print("Timing includes GA setup and GPU evaluator initialization.")

for trial_index, seed in enumerate(COMPARISON_SEEDS[:COMPARISON_TRIALS], start=1):
    # Reuse the same random seed for a paired CPU/GPU comparison.
    random.seed(seed)
    np.random.seed(seed)
    start_time = time.perf_counter()

    _, cpu_cost, _ = genetic_algorithm(
        demands=large_demands,
        vehicle_capacity=large_vehicle_capacity,
        distance_matrix=large_distance_matrix,
        population_size=COMPARISON_POPULATION,
        generations=COMPARISON_GENERATIONS,
        mutation_rate=COMPARISON_MUTATION_RATE,
        use_gpu=False
    )

    cpu_elapsed = time.perf_counter() - start_time

    random.seed(seed)
    np.random.seed(seed)
    start_time = time.perf_counter()

    _, gpu_cost, _ = genetic_algorithm(
        demands=large_demands,
        vehicle_capacity=large_vehicle_capacity,
        distance_matrix=large_distance_matrix,
        population_size=COMPARISON_POPULATION,
        generations=COMPARISON_GENERATIONS,
        mutation_rate=COMPARISON_MUTATION_RATE,
        use_gpu=True
    )

    gpu_elapsed = time.perf_counter() - start_time

    cpu_times.append(cpu_elapsed)
    gpu_times.append(gpu_elapsed)
    cpu_distances.append(cpu_cost)
    gpu_distances.append(gpu_cost)

    print(
        f"Trial {trial_index} (seed={seed}): "
        f"CPU distance={cpu_cost:.2f}, CPU time={cpu_elapsed:.4f}s | "
        f"GPU distance={gpu_cost:.2f}, GPU time={gpu_elapsed:.4f}s"
    )

average_cpu_time = float(np.mean(cpu_times))
average_gpu_time = float(np.mean(gpu_times))
average_cpu_distance = float(np.mean(cpu_distances))
average_gpu_distance = float(np.mean(gpu_distances))
speed_ratio = average_cpu_time / average_gpu_time if average_gpu_time > 0 else float("inf")

print("\nCPU vs GPU Summary:")
print("Average CPU distance:", round(average_cpu_distance, 2))
print("Average GPU distance:", round(average_gpu_distance, 2))
print("Best CPU distance:", round(min(cpu_distances), 2))
print("Best GPU distance:", round(min(gpu_distances), 2))
print("Average CPU time:", round(average_cpu_time, 4), "seconds")
print("Average GPU time:", round(average_gpu_time, 4), "seconds")
print("CPU/GPU speed ratio:", round(speed_ratio, 2), "x")

if speed_ratio > 1:
    print("Result: GPU was faster for this end-to-end benchmark.")
elif speed_ratio < 1:
    print("Result: CPU was faster for this end-to-end benchmark.")
else:
    print("Result: CPU and GPU had equal average execution time.")

print(
    "Average distance difference (GPU - CPU):",
    round(average_gpu_distance - average_cpu_distance, 2)
)
print(
    "Note: GPU fitness uses float32, while CPU calculations may use "
    "float64, so tiny fitness differences can affect GA selection."
)


# ------------------------------------------------------------
# SCALABILITY BENCHMARK: 20, 100, AND 500 CUSTOMERS
# Same GA settings and paired seeds for CPU/GPU at each size.
# ------------------------------------------------------------
print("\n" + "=" * 68)
print("CPU vs GPU SCALABILITY BENCHMARK")
print("=" * 68)
print("This section generates fresh synthetic VRP datasets.")
print("Settings per run: population=80, generations=40, trials=3.")
print("GPU buffers are reused across generations and trials.")
print("GPU setup/compilation is reported separately from warm-run timings.")

SCALING_SIZES = [20, 100, 500]
SCALING_POPULATION = 80
SCALING_GENERATIONS = 40
SCALING_TRIAL_SEEDS = [111, 222, 333]
SCALING_MUTATION_RATE = 0.2
scaling_results = []

for customer_count in SCALING_SIZES:
    # Generate deterministic coordinates and demands for this problem size.
    data_rng = np.random.default_rng(9000 + customer_count)
    locations = np.vstack((
        np.array([[50, 50]], dtype=np.int32),
        data_rng.integers(0, 101, size=(customer_count, 2), dtype=np.int32)
    ))
    demands_for_size = np.concatenate((
        np.array([0], dtype=np.int32),
        data_rng.integers(1, 5, size=customer_count, dtype=np.int32)
    ))
    capacity_for_size = 10

    # Construct the pairwise Euclidean distance matrix.
    differences = locations[:, None, :] - locations[None, :, :]
    matrix_for_size = np.sqrt(
        np.sum(differences.astype(np.float64) ** 2, axis=2)
    )

    size_cpu_times = []
    size_gpu_times = []
    size_cpu_distances = []
    size_gpu_distances = []
    cpu_solution_validity = []
    gpu_solution_validity = []

    # Compile and allocate once per dataset size, then reuse for every trial.
    gpu_setup_start = time.perf_counter()
    reusable_gpu_evaluator = create_gpu_fitness_evaluator(
        demands=demands_for_size,
        vehicle_capacity=capacity_for_size,
        distance_matrix=matrix_for_size,
        max_population_size=SCALING_POPULATION
    )
    gpu_setup_elapsed = time.perf_counter() - gpu_setup_start

    print("\n" + "-" * 68)
    print(f"Customers: {customer_count}")
    print(f"Population: {SCALING_POPULATION} | Generations: {SCALING_GENERATIONS} | Trials: {len(SCALING_TRIAL_SEEDS)}")
    print(f"GPU one-time setup/compile: {gpu_setup_elapsed:.3f}s")

    # Warm up the GPU path before timed trials.
    saved_random_state = random.getstate()
    saved_numpy_state = np.random.get_state()
    random.seed(9876)
    np.random.seed(9876)
    warm_population = create_initial_population(
        len(demands_for_size) - 1, SCALING_POPULATION
    )
    reusable_gpu_evaluator(warm_population)
    random.setstate(saved_random_state)
    np.random.set_state(saved_numpy_state)

    for trial_index, seed in enumerate(SCALING_TRIAL_SEEDS, start=1):
        random.seed(seed)
        np.random.seed(seed)
        start = time.perf_counter()
        cpu_chromosome, cpu_cost, _ = genetic_algorithm(
            demands=demands_for_size,
            vehicle_capacity=capacity_for_size,
            distance_matrix=matrix_for_size,
            population_size=SCALING_POPULATION,
            generations=SCALING_GENERATIONS,
            mutation_rate=SCALING_MUTATION_RATE,
            use_gpu=False
        )
        cpu_elapsed = time.perf_counter() - start

        random.seed(seed)
        np.random.seed(seed)
        start = time.perf_counter()
        gpu_chromosome, gpu_cost, _ = genetic_algorithm(
            demands=demands_for_size,
            vehicle_capacity=capacity_for_size,
            distance_matrix=matrix_for_size,
            population_size=SCALING_POPULATION,
            generations=SCALING_GENERATIONS,
            mutation_rate=SCALING_MUTATION_RATE,
            use_gpu=True,
            gpu_evaluator=reusable_gpu_evaluator
        )
        gpu_elapsed = time.perf_counter() - start

        size_cpu_times.append(cpu_elapsed)
        size_gpu_times.append(gpu_elapsed)
        size_cpu_distances.append(float(cpu_cost))
        size_gpu_distances.append(float(gpu_cost))

        # Check every customer is served exactly once and vehicle capacities hold.
        cpu_routes = decode_chromosome(
            cpu_chromosome, demands_for_size, capacity_for_size
        )
        gpu_routes = decode_chromosome(
            gpu_chromosome, demands_for_size, capacity_for_size
        )

        def _is_valid_vrp_solution(routes, demands_array, capacity):
            served = [customer for route in routes for customer in route if customer != 0]
            expected = list(range(1, len(demands_array)))
            coverage_ok = sorted(served) == expected
            capacity_ok = all(
                sum(int(demands_array[customer]) for customer in route if customer != 0)
                <= capacity
                for route in routes
            )
            return coverage_ok and capacity_ok

        cpu_solution_validity.append(
            _is_valid_vrp_solution(cpu_routes, demands_for_size, capacity_for_size)
        )
        gpu_solution_validity.append(
            _is_valid_vrp_solution(gpu_routes, demands_for_size, capacity_for_size)
        )

        print(
            f"Trial {trial_index}: CPU={cpu_elapsed:.3f}s, GPU={gpu_elapsed:.3f}s, "
            f"CPU distance={cpu_cost:.2f}, GPU distance={gpu_cost:.2f}"
        )

    avg_cpu = float(np.mean(size_cpu_times))
    avg_gpu = float(np.mean(size_gpu_times))
    median_cpu = float(np.median(size_cpu_times))
    median_gpu = float(np.median(size_gpu_times))
    speedup = avg_cpu / avg_gpu if avg_gpu > 0 else float("inf")
    time_reduction_pct = ((avg_cpu - avg_gpu) / avg_cpu * 100.0) if avg_cpu > 0 else 0.0
    avg_cpu_dist = float(np.mean(size_cpu_distances))
    avg_gpu_dist = float(np.mean(size_gpu_distances))

    scaling_results.append({
        "customers": customer_count,
        "trials": len(SCALING_TRIAL_SEEDS),
        "population_size": SCALING_POPULATION,
        "generations": SCALING_GENERATIONS,
        "gpu_setup_seconds": gpu_setup_elapsed,
        "cpu_avg_seconds": avg_cpu,
        "gpu_avg_seconds": avg_gpu,
        "cpu_median_seconds": median_cpu,
        "gpu_median_seconds": median_gpu,
        "cpu_min_seconds": float(np.min(size_cpu_times)),
        "gpu_min_seconds": float(np.min(size_gpu_times)),
        "cpu_max_seconds": float(np.max(size_cpu_times)),
        "gpu_max_seconds": float(np.max(size_gpu_times)),
        "speedup": speedup,
        "gpu_time_reduction_percent": time_reduction_pct,
        "cpu_average_distance": avg_cpu_dist,
        "gpu_average_distance": avg_gpu_dist,
        "average_distance_difference_gpu_minus_cpu": avg_gpu_dist - avg_cpu_dist,
        "cpu_all_solutions_valid": all(cpu_solution_validity),
        "gpu_all_solutions_valid": all(gpu_solution_validity),
    })

    print(f"Average CPU time: {avg_cpu:.3f}s")
    print(f"Average GPU time: {avg_gpu:.3f}s")
    print(f"CPU/GPU speed ratio: {speedup:.2f}x")
    print(f"Average distance difference (GPU - CPU): {avg_gpu_dist - avg_cpu_dist:.4f}")
    if speedup > 1.02:
        print("Result: GPU faster on average.")
    elif speedup < 0.98:
        print("Result: CPU faster on average.")
    else:
        print("Result: approximately tied; difference is within 2%.")

print("\n" + "=" * 68)
print("SCALABILITY SUMMARY")
print("=" * 68)
print(f"{'Customers':>10} | {'CPU avg (s)':>12} | {'GPU avg (s)':>12} | {'Speedup':>9} | {'CPU dist':>10} | {'GPU dist':>10}")
print("-" * 78)
for result in scaling_results:
    print(
        f"{result['customers']:>10} | {result['cpu_avg_seconds']:>12.3f} | "
        f"{result['gpu_avg_seconds']:>12.3f} | {result['speedup']:>8.2f}x | "
        f"{result['cpu_average_distance']:>10.2f} | {result['gpu_average_distance']:>10.2f}"
    )
print("\nInterpretation: speedup > 1.00x means GPU was faster; below 1.00x means CPU was faster.")
print("GPU trial timings exclude one-time setup/compilation, reported separately per dataset.")
print("These are synthetic benchmark results from this machine, not universal performance guarantees.")

# ------------------------------------------------------------
# EXPORT BENCHMARK REPORT: CSV + GRAPH + VALIDATION SUMMARY
# ------------------------------------------------------------
output_dir = Path(__file__).resolve().parent
csv_path = output_dir / "vrp_benchmark_results.csv"
graph_path = output_dir / "vrp_benchmark_comparison.png"

if scaling_results:
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(scaling_results[0].keys()))
        writer.writeheader()
        writer.writerows(scaling_results)

    print("\n" + "=" * 68)
    print("BENCHMARK REPORT DETAILS")
    print("=" * 68)
    print(f"CSV report saved to: {csv_path}")
    print(f"Comparison graph saved to: {graph_path}")
    print("\nDetailed metrics:")
    print(f"{'Customers':>10} | {'CPU median':>10} | {'GPU median':>10} | {'GPU time saved':>14} | {'CPU valid':>10} | {'GPU valid':>10}")
    print("-" * 84)
    for result in scaling_results:
        print(
            f"{result['customers']:>10} | {result['cpu_median_seconds']:>9.3f}s | "
            f"{result['gpu_median_seconds']:>9.3f}s | "
            f"{result['gpu_time_reduction_percent']:>13.2f}% | "
            f"{str(result['cpu_all_solutions_valid']):>10} | "
            f"{str(result['gpu_all_solutions_valid']):>10}"
        )

    # Save the graph without opening a blocking plot window.
    import matplotlib.pyplot as plt

    customer_labels = [str(result["customers"]) for result in scaling_results]
    cpu_medians = [result["cpu_median_seconds"] for result in scaling_results]
    gpu_medians = [result["gpu_median_seconds"] for result in scaling_results]
    x = np.arange(len(customer_labels))
    width = 0.36

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.bar(x - width / 2, cpu_medians, width, label="CPU median")
    ax.bar(x + width / 2, gpu_medians, width, label="GPU median")
    ax.set_title("VRP CPU vs GPU Median Execution Time")
    ax.set_xlabel("Number of customers")
    ax.set_ylabel("Execution time (seconds; lower is better)")
    ax.set_xticks(x)
    ax.set_xticklabels(customer_labels)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    fig.tight_layout()
    fig.savefig(graph_path, dpi=160)
    plt.close(fig)

    print("\nReport files generated. Review the CSV for all metrics and the PNG for the timing comparison.")
