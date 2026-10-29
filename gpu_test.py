
import pyopencl as cl
import numpy as np

# Find the Intel GPU
platform = cl.get_platforms()[0]
gpu = platform.get_devices(device_type=cl.device_type.GPU)[0]

print("GPU detected:", gpu.name)

# Create an OpenCL context and command queue
context = cl.Context([gpu])
queue = cl.CommandQueue(context)

# GPU program: add two arrays
kernel_code = """
__kernel void vector_add(
    __global const float *a,
    __global const float *b,
    __global float *result
)
{
    int i = get_global_id(0);
    result[i] = a[i] + b[i];
}
"""

program = cl.Program(context, kernel_code).build()

# Input arrays
a = np.array([10, 20, 30, 40], dtype=np.float32)
b = np.array([1, 2, 3, 4], dtype=np.float32)
result = np.empty_like(a)

# Copy data to GPU memory
mf = cl.mem_flags
a_gpu = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a)
b_gpu = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b)
result_gpu = cl.Buffer(context, mf.WRITE_ONLY, result.nbytes)

# Execute the kernel on the GPU
program.vector_add(
    queue, a.shape, None, a_gpu, b_gpu, result_gpu
)

# Copy results back to CPU
cl.enqueue_copy(queue, result, result_gpu).wait()

print("Array A:", a)
print("Array B:", b)
print("GPU result:", result)
print("Expected:  ", np.array([11, 22, 33, 44]))
print("GPU test passed:", np.array_equal(result, a + b))
