"""Indigenous Hybrid CPU-GPU Optimization Solver for Logistics.

Run directly to execute the full research/benchmark workflow.
Import this module from dashboard_app.py to reuse the same solver functions
without automatically running all experiments.
"""

import time

import csv

from pathlib import Path

import numpy as np

import random

import matplotlib.pyplot as plt


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


def create_initial_population(num_customers, population_size):
    """Create random customer orderings."""
    population = []

    customers = list(range(1, num_customers + 1))

    for _ in range(population_size):
        chromosome = customers.copy()
        random.shuffle(chromosome)
        population.append(chromosome)

    return population


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


def mutate_chromosome(chromosome, mutation_rate=0.2):
    """Randomly swap two customers with a given probability."""
    mutated = chromosome.copy()

    if len(mutated) >= 2 and random.random() < mutation_rate:
        i, j = random.sample(range(len(mutated)), 2)
        mutated[i], mutated[j] = mutated[j], mutated[i]

    return mutated


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


def select_backend(customer_count, population_size):
    """Choose a backend from the latest measured benchmark combinations.

    Exact combinations are based on the latest adaptive sweep, plus the
    targeted 20-customer/population-1000 benchmark. Unmeasured combinations
    default to CPU rather than extrapolating from noisy benchmark results.
    Re-benchmark on other hardware.
    """
    measured_recommendations = {
        # Latest adaptive sweep results on the current Intel UHD Graphics system.
        (20, 20): "CPU",
        (20, 80): "CPU",
        (20, 160): "CPU",
        (20, 320): "CPU",
        (100, 20): "CPU",
        (100, 80): "GPU",
        (100, 160): "GPU",
        (100, 320): "GPU",
        (500, 20): "CPU",
        (500, 80): "GPU",
        (500, 160): "CPU",
        (500, 320): "GPU",
        # Separate targeted benchmark: 20 customers, population 1000.
        # Median CPU=0.316s, GPU=0.286s, speedup=1.11x.
        (20, 1000): "GPU",
    }
    return measured_recommendations.get((customer_count, population_size), "CPU")


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

    # use_gpu may be True, False, or "auto". Auto mode uses the empirical
    # workload policy discovered by the adaptive benchmark.
    if use_gpu == "auto":
        selected_backend = select_backend(num_customers, population_size)
    else:
        selected_backend = "GPU" if use_gpu else "CPU"

    gpu_active = selected_backend == "GPU"
    if gpu_active and gpu_evaluator is None:
        try:
            gpu_evaluator = create_gpu_fitness_evaluator(
                demands,
                vehicle_capacity,
                distance_matrix,
                max_population_size=population_size
            )
        except Exception as exc:
            print(f"GPU initialization failed ({exc}); falling back to CPU.")
            gpu_active = False
            selected_backend = "CPU"

    print(
        f"Adaptive backend: {selected_backend} "
        f"(customers={num_customers}, population={population_size})"
    )

    best_chromosome = None
    best_cost = float("inf")
    history = []

    for generation in range(generations):
        # Evaluate the whole population on the selected backend. If GPU
        # execution fails mid-run, switch safely to CPU for this and later generations.
        if gpu_active:
            try:
                population_costs = gpu_evaluator(population)
            except Exception as exc:
                print(f"GPU evaluation failed ({exc}); switching to CPU.")
                gpu_active = False
                selected_backend = "CPU"
                population_costs = [
                    calculate_fitness(
                        chromosome,
                        demands,
                        vehicle_capacity,
                        distance_matrix
                    )
                    for chromosome in population
                ]
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


def _is_valid_vrp_solution(routes, demands_array, capacity):
    served = [customer for route in routes for customer in route if customer != 0]
    expected = list(range(1, len(demands_array)))
    coverage_ok = sorted(served) == expected
    capacity_ok = all(
        sum(int(demands_array[customer]) for customer in route if customer != 0) <= capacity
        for route in routes
    )
    return coverage_ok and capacity_ok


def main():
    print("Indigenous Hybrid CPU-GPU Optimization Solver")

    print("Domain: Logistics")

    print("Problem: Vehicle Routing Problem (VRP)")

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

    difference = locations[:, np.newaxis, :] - locations[np.newaxis, :, :]

    distance_matrix = np.sqrt(np.sum(difference ** 2, axis=2))

    print("\nDistance Matrix:")

    print(np.round(distance_matrix, 2))

    demands = np.array([0, 2, 3, 2, 4])

    vehicle_capacity = 6

    print("\nCustomer Demands:")

    for i in range(1, len(locations)):
        print(f"Customer {i}: {demands[i]} units")

    print("\nVehicle capacity:", vehicle_capacity)

    print("Total customer demand:", np.sum(demands))

    print("Minimum vehicles required (capacity only):",
          int(np.ceil(np.sum(demands) / vehicle_capacity)))

    route = [0, 1, 2, 3, 4, 0]

    total_distance = 0

    for i in range(len(route) - 1):
        current_location = route[i]
        next_location = route[i + 1]

        total_distance += distance_matrix[
            current_location, next_location
        ]

    print("\nSample Route:", " -> ".join(map(str, route)))

    print("Total route distance:", round(total_distance, 2))

    route_customers = route[1:-1]

    route_demand = sum(demands[customer] for customer in route_customers)

    print("\nRoute demand:", route_demand)

    print("Vehicle capacity:", vehicle_capacity)

    if route_demand <= vehicle_capacity:
        print("Route is FEASIBLE: capacity constraint satisfied.")
    else:
        print("Route is INFEASIBLE: vehicle capacity exceeded.")

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

    visited_customers = []

    for vehicle_route in routes:
        # Exclude the starting and ending warehouse
        customers = vehicle_route[1:-1]
        visited_customers.extend(customers)

    all_customers = list(range(1, len(locations)))

    customers_visited_once = (
        sorted(visited_customers) == all_customers
    )

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

    test_chromosome = [1, 2, 3, 4]

    decoded_routes = decode_chromosome(
        test_chromosome, demands, vehicle_capacity
    )

    print("\nChromosome Decoder Test:")

    print("Chromosome:", test_chromosome)

    print("Decoded routes:", decoded_routes)

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

    population_size = 10

    population = create_initial_population(
        len(locations) - 1,
        population_size
    )

    print("\nInitial Genetic Algorithm Population:")

    for index, chromosome in enumerate(population, start=1):
        print(f"Chromosome {index}: {chromosome}")

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

    best_index = population_costs.index(min(population_costs))

    best_chromosome = population[best_index]

    best_cost = population_costs[best_index]

    print("\nBest Candidate:")

    print("Chromosome:", best_chromosome)

    print("Total distance:", round(best_cost, 2))

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

    parent1 = [1, 2, 3, 4]

    parent2 = [3, 4, 1, 2]

    child = order_crossover(parent1, parent2)

    print("\nCrossover Test:")

    print("Parent 1:", parent1)

    print("Parent 2:", parent2)

    print("Child:", child)

    print("Child contains every customer once:", sorted(child) == [1, 2, 3, 4])

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

    print("Adaptive CPU/GPU backend selection is enabled for this GA run.")

    print("Number of customers:", len(large_locations) - 1)

    print("Number of vehicles needed at minimum:", 
          int(np.ceil(large_demands[1:].sum() / large_vehicle_capacity)))

    print("Total customer demand:", large_demands[1:].sum())

    large_ga_chromosome, large_ga_cost, large_history = genetic_algorithm(
        demands=large_demands,
        vehicle_capacity=large_vehicle_capacity,
        distance_matrix=large_distance_matrix,
        population_size=1000,
        generations=10,
        mutation_rate=0.2,
        use_gpu="auto"
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

    large_greedy_distance = sum(
        calculate_route_distance(route, large_distance_matrix)
        for route in large_greedy_routes
    )

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

    plt.figure(figsize=(10, 8))

    plt.scatter(
        large_locations[0, 0],
        large_locations[0, 1],
        color="red",
        marker="s",
        s=150,
        label="Warehouse",
        zorder=5
    )

    plt.scatter(
        large_locations[1:, 0],
        large_locations[1:, 1],
        color="blue",
        marker="o",
        s=60,
        label="Customers"
    )

    for i, (x, y) in enumerate(large_locations):
        plt.annotate(
            str(i),
            (x, y),
            xytext=(5, 5),
            textcoords="offset points"
        )

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

    print("\n" + "=" * 72)

    print("ADAPTIVE CPU/GPU POPULATION-SIZE BENCHMARK")

    print("=" * 72)

    print("This benchmark tests several GA population sizes on fixed synthetic VRPs.")

    print("It measures warm-run CPU/GPU time with paired random seeds.")

    print("It recommends a backend per tested workload; it does not assume a universal threshold.")

    ADAPTIVE_CUSTOMER_SIZES = [20, 100, 500]

    ADAPTIVE_POPULATION_SIZES = [20, 80, 160, 320]

    ADAPTIVE_GENERATIONS = 20

    ADAPTIVE_TRIAL_SEEDS = [111, 222]

    ADAPTIVE_MUTATION_RATE = 0.2

    adaptive_results = []

    for customer_count in ADAPTIVE_CUSTOMER_SIZES:
        data_rng = np.random.default_rng(9000 + customer_count)
        locations_for_size = np.vstack((
            np.array([[50, 50]], dtype=np.int32),
            data_rng.integers(0, 101, size=(customer_count, 2), dtype=np.int32)
        ))
        demands_for_size = np.concatenate((
            np.array([0], dtype=np.int32),
            data_rng.integers(1, 5, size=customer_count, dtype=np.int32)
        ))
        capacity_for_size = 10
        differences = locations_for_size[:, None, :] - locations_for_size[None, :, :]
        matrix_for_size = np.sqrt(np.sum(differences.astype(np.float64) ** 2, axis=2))
        max_population = max(ADAPTIVE_POPULATION_SIZES)

        print("\n" + "-" * 72)
        print(f"Customers: {customer_count}")
        print(f"Population sizes: {ADAPTIVE_POPULATION_SIZES}; generations: {ADAPTIVE_GENERATIONS}; trials per size: {len(ADAPTIVE_TRIAL_SEEDS)}")

        setup_start = time.perf_counter()
        gpu_evaluator = create_gpu_fitness_evaluator(
            demands=demands_for_size,
            vehicle_capacity=capacity_for_size,
            distance_matrix=matrix_for_size,
            max_population_size=max_population
        )
        gpu_setup_seconds = time.perf_counter() - setup_start
        print(f"GPU setup/compilation (excluded from timed runs): {gpu_setup_seconds:.3f}s")

        for population_size in ADAPTIVE_POPULATION_SIZES:
            cpu_times, gpu_times = [], []
            cpu_distances, gpu_distances = [], []
            cpu_valid, gpu_valid = [], []

            # Warm the GPU evaluator for this batch size before timing.
            warm_population = create_initial_population(customer_count, population_size)
            gpu_evaluator(warm_population)

            for seed in ADAPTIVE_TRIAL_SEEDS:
                random.seed(seed)
                np.random.seed(seed)
                start = time.perf_counter()
                cpu_chromosome, cpu_cost, _ = genetic_algorithm(
                    demands=demands_for_size,
                    vehicle_capacity=capacity_for_size,
                    distance_matrix=matrix_for_size,
                    population_size=population_size,
                    generations=ADAPTIVE_GENERATIONS,
                    mutation_rate=ADAPTIVE_MUTATION_RATE,
                    use_gpu=False
                )
                cpu_times.append(time.perf_counter() - start)
                cpu_distances.append(float(cpu_cost))
                cpu_routes = decode_chromosome(cpu_chromosome, demands_for_size, capacity_for_size)
                cpu_valid.append(_is_valid_vrp_solution(cpu_routes, demands_for_size, capacity_for_size))

                random.seed(seed)
                np.random.seed(seed)
                start = time.perf_counter()
                gpu_chromosome, gpu_cost, _ = genetic_algorithm(
                    demands=demands_for_size,
                    vehicle_capacity=capacity_for_size,
                    distance_matrix=matrix_for_size,
                    population_size=population_size,
                    generations=ADAPTIVE_GENERATIONS,
                    mutation_rate=ADAPTIVE_MUTATION_RATE,
                    use_gpu=True,
                    gpu_evaluator=gpu_evaluator
                )
                gpu_times.append(time.perf_counter() - start)
                gpu_distances.append(float(gpu_cost))
                gpu_routes = decode_chromosome(gpu_chromosome, demands_for_size, capacity_for_size)
                gpu_valid.append(_is_valid_vrp_solution(gpu_routes, demands_for_size, capacity_for_size))

            cpu_median = float(np.median(cpu_times))
            gpu_median = float(np.median(gpu_times))
            speedup = cpu_median / gpu_median if gpu_median > 0 else float("inf")
            # Choose GPU only if the observed median advantage exceeds 2%.
            recommendation = "GPU" if speedup > 1.02 else "CPU"
            row = {
                "customers": customer_count,
                "population_size": population_size,
                "generations": ADAPTIVE_GENERATIONS,
                "trials": len(ADAPTIVE_TRIAL_SEEDS),
                "gpu_setup_seconds_excluded": gpu_setup_seconds,
                "cpu_median_seconds": cpu_median,
                "gpu_median_seconds": gpu_median,
                "cpu_average_seconds": float(np.mean(cpu_times)),
                "gpu_average_seconds": float(np.mean(gpu_times)),
                "cpu_over_gpu_speedup": speedup,
                "gpu_time_reduction_percent": ((cpu_median - gpu_median) / cpu_median * 100.0) if cpu_median else 0.0,
                "cpu_average_distance": float(np.mean(cpu_distances)),
                "gpu_average_distance": float(np.mean(gpu_distances)),
                "cpu_all_solutions_valid": all(cpu_valid),
                "gpu_all_solutions_valid": all(gpu_valid),
                "recommended_backend_for_tested_size": recommendation,
            }
            adaptive_results.append(row)
            print(
                f"Population {population_size:>3}: CPU median={cpu_median:.3f}s | "
                f"GPU median={gpu_median:.3f}s | speedup={speedup:.2f}x | "
                f"recommended={recommendation} | valid CPU/GPU={all(cpu_valid)}/{all(gpu_valid)}"
            )

    output_dir = Path(__file__).resolve().parent

    csv_path = output_dir / "vrp_adaptive_population_results.csv"

    graph_path = output_dir / "vrp_adaptive_population_comparison.png"

    if adaptive_results:
        with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=list(adaptive_results[0].keys()))
            writer.writeheader()
            writer.writerows(adaptive_results)

        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, len(ADAPTIVE_CUSTOMER_SIZES), figsize=(15, 4.8), sharey=False)
        if len(ADAPTIVE_CUSTOMER_SIZES) == 1:
            axes = [axes]
        for ax, customer_count in zip(axes, ADAPTIVE_CUSTOMER_SIZES):
            rows = [r for r in adaptive_results if r["customers"] == customer_count]
            x = np.arange(len(rows))
            width = 0.36
            ax.bar(x - width / 2, [r["cpu_median_seconds"] for r in rows], width, label="CPU median")
            ax.bar(x + width / 2, [r["gpu_median_seconds"] for r in rows], width, label="GPU median")
            ax.set_title(f"{customer_count} customers")
            ax.set_xlabel("Population size")
            ax.set_xticks(x)
            ax.set_xticklabels([str(r["population_size"]) for r in rows])
            ax.grid(axis="y", linestyle="--", alpha=0.3)
        axes[0].set_ylabel("Median execution time (seconds)")
        axes[-1].legend()
        fig.suptitle("CPU vs GPU by GA Population Size (lower is better)")
        fig.tight_layout()
        fig.savefig(graph_path, dpi=160)
        plt.close(fig)

        print("\n" + "=" * 72)
        print("ADAPTIVE BACKEND RECOMMENDATIONS")
        print("=" * 72)
        print(f"{'Customers':>10} | {'Population':>10} | {'Speedup':>8} | {'Recommended':>12} | {'Valid CPU/GPU':>14}")
        print("-" * 66)
        for row in adaptive_results:
            print(
                f"{row['customers']:>10} | {row['population_size']:>10} | "
                f"{row['cpu_over_gpu_speedup']:>7.2f}x | "
                f"{row['recommended_backend_for_tested_size']:>12} | "
                f"{str(row['cpu_all_solutions_valid']) + '/' + str(row['gpu_all_solutions_valid']):>14}"
            )
        print(f"\nCSV report: {csv_path}")
        print(f"Comparison graph: {graph_path}")
        print("Policy: GPU is recommended only when its median time is at least 2% lower than CPU.")
        print("Recommendations apply only to these tested workloads and this machine; rerun to confirm noisy results.")

    TARGET_CUSTOMERS = len(large_demands) - 1

    TARGET_POPULATION = 1000

    TARGET_GENERATIONS = 10

    TARGET_TRIAL_SEEDS = [101, 202, 303, 404, 505]

    TARGET_MUTATION_RATE = 0.2

    target_cpu_times = []

    target_gpu_times = []

    target_cpu_distances = []

    target_gpu_distances = []

    target_validity = []

    print("\n" + "=" * 72)

    print("TARGETED CPU vs GPU BENCHMARK: MAIN WORKLOAD")

    print("=" * 72)

    print(f"Customers: {TARGET_CUSTOMERS}")

    print(f"Population size: {TARGET_POPULATION}")

    print(f"Generations: {TARGET_GENERATIONS}")

    print(f"Paired trials: {len(TARGET_TRIAL_SEEDS)}")

    print("Timing includes GA setup and GPU evaluator initialization.")

    for trial_number, seed in enumerate(TARGET_TRIAL_SEEDS, start=1):
        random.seed(seed)
        np.random.seed(seed)
        start = time.perf_counter()
        cpu_chromosome, cpu_cost, _ = genetic_algorithm(
            demands=large_demands,
            vehicle_capacity=large_vehicle_capacity,
            distance_matrix=large_distance_matrix,
            population_size=TARGET_POPULATION,
            generations=TARGET_GENERATIONS,
            mutation_rate=TARGET_MUTATION_RATE,
            use_gpu=False,
        )
        cpu_elapsed = time.perf_counter() - start

        random.seed(seed)
        np.random.seed(seed)
        start = time.perf_counter()
        gpu_chromosome, gpu_cost, _ = genetic_algorithm(
            demands=large_demands,
            vehicle_capacity=large_vehicle_capacity,
            distance_matrix=large_distance_matrix,
            population_size=TARGET_POPULATION,
            generations=TARGET_GENERATIONS,
            mutation_rate=TARGET_MUTATION_RATE,
            use_gpu=True,
        )
        gpu_elapsed = time.perf_counter() - start

        # Validate decoded routes from both solutions.
        cpu_routes = decode_chromosome(cpu_chromosome, large_demands, large_vehicle_capacity)
        gpu_routes = decode_chromosome(gpu_chromosome, large_demands, large_vehicle_capacity)
        cpu_valid = _is_valid_vrp_solution(cpu_routes, large_demands, large_vehicle_capacity)
        gpu_valid = _is_valid_vrp_solution(gpu_routes, large_demands, large_vehicle_capacity)

        target_cpu_times.append(cpu_elapsed)
        target_gpu_times.append(gpu_elapsed)
        target_cpu_distances.append(float(cpu_cost))
        target_gpu_distances.append(float(gpu_cost))
        target_validity.append((cpu_valid, gpu_valid))

        print(
            f"Trial {trial_number} (seed={seed}): "
            f"CPU={cpu_elapsed:.3f}s, distance={cpu_cost:.2f}, valid={cpu_valid} | "
            f"GPU={gpu_elapsed:.3f}s, distance={gpu_cost:.2f}, valid={gpu_valid}"
        )

    cpu_median = float(np.median(target_cpu_times))

    gpu_median = float(np.median(target_gpu_times))

    target_speedup = cpu_median / gpu_median if gpu_median > 0 else float("inf")

    recommendation = "GPU" if gpu_median <= cpu_median * 0.98 else "CPU"

    all_target_valid = all(cpu_ok and gpu_ok for cpu_ok, gpu_ok in target_validity)

    print("\nTARGETED BENCHMARK SUMMARY")

    print(f"CPU median time: {cpu_median:.3f}s")

    print(f"GPU median time: {gpu_median:.3f}s")

    print(f"CPU/GPU median speedup: {target_speedup:.2f}x")

    print(f"Median CPU distance: {np.median(target_cpu_distances):.2f}")

    print(f"Median GPU distance: {np.median(target_gpu_distances):.2f}")

    print(f"All CPU/GPU solutions valid: {all_target_valid}")

    print(f"Recommendation for this tested workload: {recommendation}")

    print("Note: repeat on another run; results are hardware- and workload-specific.")



if __name__ == "__main__":
    main()
