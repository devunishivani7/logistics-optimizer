
import time
import numpy as np
import pyopencl as cl

# ============================================================
# 1. CREATE THE 20-CUSTOMER DATASET
# ============================================================

np.random.seed(42)

locations = np.vstack((
    np.array([[50, 50]]),
    np.random.randint(0, 101, size=(20, 2))
)).astype(np.float32)

demands = np.concatenate((
    np.array([0]),
    np.random.randint(1, 5, size=20)
))

vehicle_capacity = 10
n = len(locations)

# Calculate the distance matrix on the CPU
delta = locations[:, None, :] - locations[None, :, :]
distance_matrix = np.sqrt(
    np.sum(delta * delta, axis=2)
).astype(np.float32)

# Ensure contiguous memory for OpenCL
flat_distances = np.ascontiguousarray(
    distance_matrix
).reshape(-1)


# ============================================================
# 2. CPU ROUTE DECODING AND FITNESS
# ============================================================

def decode_chromosome(chromosome, demands, vehicle_capacity):
    """Split a customer ordering into capacity-feasible routes."""
    routes = []
    route = [0]
    load = 0

    for customer in chromosome:
        demand = demands[customer]

        if load + demand > vehicle_capacity:
            route.append(0)
            routes.append(route)

            route = [0]
            load = 0

        route.append(int(customer))
        load += demand

    route.append(0)
    routes.append(route)

    return routes


def calculate_fitness_cpu(
    chromosome, demands, vehicle_capacity, distance_matrix
):
    """Calculate total fleet distance for one chromosome."""
    routes = decode_chromosome(
        chromosome, demands, vehicle_capacity
    )

    total = 0.0

    for route in routes:
        for i in range(len(route) - 1):
            total += distance_matrix[
                route[i], route[i + 1]
            ]

    return total


# ============================================================
# 3. OPENCL GPU KERNEL
# One GPU work-item evaluates one chromosome
# ============================================================

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


# ============================================================
# 4. INITIALIZE GPU
# ============================================================

platforms = cl.get_platforms()

gpu_devices = []

for platform in platforms:
    gpu_devices.extend(
        platform.get_devices(
            device_type=cl.device_type.GPU
        )
    )

if not gpu_devices:
    raise RuntimeError(
        "No OpenCL GPU found. Check your graphics driver."
    )

gpu = gpu_devices[0]
context = cl.Context([gpu])
queue = cl.CommandQueue(context)

program = cl.Program(
    context, kernel_code
).build()

# Reuse one kernel object to avoid RepeatedKernelRetrieval
evaluate_routes_kernel = cl.Kernel(
    program, "evaluate_routes"
)

mf = cl.mem_flags

print("GPU:", gpu.name)


# ============================================================
# 5. GENERATE 1,000 CUSTOMER PERMUTATIONS
# ============================================================

population_size = 1000

population = [
    np.random.permutation(
        np.arange(1, n)
    ).tolist()
    for _ in range(population_size)
]


# ============================================================
# 6. PREPARE ROUTES FOR GPU EVALUATION
# ============================================================

def prepare_gpu_routes(population):
    """Decode chromosomes and create padded route arrays."""
    decoded_routes = [
        sum(
            decode_chromosome(
                chromosome,
                demands,
                vehicle_capacity
            ),
            []
        )
        for chromosome in population
    ]

    max_stops = max(
        len(route) for route in decoded_routes
    )

    route_lengths = np.array(
        [len(route) for route in decoded_routes],
        dtype=np.int32
    )

    route_data = np.zeros(
        (len(population), max_stops),
        dtype=np.int32
    )

    for i, route in enumerate(decoded_routes):
        route_data[i, :len(route)] = route

    return route_data, route_lengths, max_stops


def evaluate_gpu(
    route_data,
    route_lengths,
    max_stops,
    distances_buffer
):
    """Run GPU fitness evaluation and return results."""
    batch_size = len(route_lengths)

    route_buffer = cl.Buffer(
        context,
        mf.READ_ONLY | mf.COPY_HOST_PTR,
        hostbuf=route_data
    )

    length_buffer = cl.Buffer(
        context,
        mf.READ_ONLY | mf.COPY_HOST_PTR,
        hostbuf=route_lengths
    )

    fitness_buffer = cl.Buffer(
        context,
        mf.WRITE_ONLY,
        batch_size * np.dtype(np.float32).itemsize
    )

    evaluate_routes_kernel(
        queue,
        (batch_size,),
        None,
        route_buffer,
        length_buffer,
        distances_buffer,
        fitness_buffer,
        np.int32(n),
        np.int32(max_stops)
    )

    results = np.empty(
        batch_size,
        dtype=np.float32
    )

    cl.enqueue_copy(
        queue, results, fitness_buffer
    ).wait()

    return results


# Distance matrix is transferred once, outside the benchmark
distance_buffer = cl.Buffer(
    context,
    mf.READ_ONLY | mf.COPY_HOST_PTR,
    hostbuf=flat_distances
)

route_data, route_lengths, max_stops = (
    prepare_gpu_routes(population)
)


# ============================================================
# 7. VERIFY CPU AND GPU RESULTS
# ============================================================

cpu_fitness = np.array([
    calculate_fitness_cpu(
        chromosome,
        demands,
        vehicle_capacity,
        distance_matrix
    )
    for chromosome in population
])

gpu_fitness = evaluate_gpu(
    route_data,
    route_lengths,
    max_stops,
    distance_buffer
)

queue.finish()

print("\nCorrectness Test")
print("Chromosomes evaluated:", population_size)
print(
    "First five CPU fitness values:",
    np.round(cpu_fitness[:5], 2)
)
print(
    "First five GPU fitness values:",
    np.round(gpu_fitness[:5], 2)
)

results_match = np.allclose(
    cpu_fitness,
    gpu_fitness,
    atol=1e-3
)

print("CPU and GPU fitness match:", results_match)


# ============================================================
# 8. WARM-UP
# ============================================================

for chromosome in population[:10]:
    calculate_fitness_cpu(
        chromosome,
        demands,
        vehicle_capacity,
        distance_matrix
    )

evaluate_gpu(
    route_data,
    route_lengths,
    max_stops,
    distance_buffer
)

queue.finish()


# ============================================================
# 9. BENCHMARK CPU FITNESS
# ============================================================

cpu_times = []

for _ in range(30):
    start = time.perf_counter()

    cpu_results = np.array([
        calculate_fitness_cpu(
            chromosome,
            demands,
            vehicle_capacity,
            distance_matrix
        )
        for chromosome in population
    ])

    cpu_times.append(
        time.perf_counter() - start
    )


# ============================================================
# 10. BENCHMARK GPU END-TO-END
# Includes route preparation, buffers, execution and result copy
# Distance matrix transfer is excluded because it is reused.
# ============================================================

gpu_times = []

for _ in range(30):
    start = time.perf_counter()

    batch_data, batch_lengths, batch_max_stops = (
        prepare_gpu_routes(population)
    )

    batch_result = evaluate_gpu(
        batch_data,
        batch_lengths,
        batch_max_stops,
        distance_buffer
    )

    gpu_times.append(
        time.perf_counter() - start
    )


# ============================================================
# 11. DISPLAY BENCHMARK RESULTS
# ============================================================

cpu_median = np.median(cpu_times)
gpu_median = np.median(gpu_times)

print("\nFitness Benchmark")
print("Chromosomes per batch:", population_size)
print(f"CPU median: {cpu_median * 1000:.3f} ms")
print(
    f"GPU end-to-end median: "
    f"{gpu_median * 1000:.3f} ms"
)

if gpu_median > 0:
    print(
        f"CPU/GPU speed ratio: "
        f"{cpu_median / gpu_median:.2f}x"
    )

print(
    "Benchmark results match:",
    np.allclose(
        cpu_results,
        batch_result,
        atol=1e-3
    )
)
