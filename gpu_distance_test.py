
import numpy as np
import pyopencl as cl

# Sample warehouse and customer coordinates
locations = np.array([
    [50, 50],  # Warehouse
    [10, 20],
    [30, 40],
    [60, 70],
    [80, 10]
], dtype=np.float32)

# Set up Intel GPU
platform = cl.get_platforms()[0]
gpu = platform.get_devices(device_type=cl.device_type.GPU)[0]
context = cl.Context([gpu])
queue = cl.CommandQueue(context)

# GPU kernel: calculate Euclidean distances
kernel_code = """
__kernel void calculate_distances(
    __global const float2 *locations,
    __global float *distances,
    const int n
)
{
    int i = get_global_id(0);
    int j = get_global_id(1);

    if (i < n && j < n) {
        float2 a = locations[i];
        float2 b = locations[j];

        float dx = a.x - b.x;
        float dy = a.y - b.y;

        distances[i * n + j] = sqrt(dx * dx + dy * dy);
    }
}
"""

program = cl.Program(context, kernel_code).build()

n = len(locations)
flat_locations = np.ascontiguousarray(locations).reshape(-1)
distances = np.empty((n, n), dtype=np.float32)

mf = cl.mem_flags

locations_gpu = cl.Buffer(
    context, mf.READ_ONLY | mf.COPY_HOST_PTR,
    hostbuf=flat_locations
)
distances_gpu = cl.Buffer(
    context, mf.WRITE_ONLY, distances.nbytes
)

# Launch one GPU work-item for each pair of locations
program.calculate_distances(
    queue, (n, n), None,
    locations_gpu, distances_gpu, np.int32(n)
)

cl.enqueue_copy(queue, distances, distances_gpu).wait()

print("GPU:", gpu.name)
print("\nGPU distance matrix:")
print(np.round(distances, 2))

# Verify against NumPy's CPU calculation
differences = locations[:, None, :] - locations[None, :, :]
cpu_distances = np.sqrt(np.sum(differences ** 2, axis=2))

print("\nCPU and GPU results match:",
      np.allclose(distances, cpu_distances, atol=1e-4))
print("Diagonal is zero:",
      np.allclose(np.diag(distances), 0))
print("Matrix is symmetric:",
      np.allclose(distances, distances.T, atol=1e-4))
