#include <iostream>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <curand_kernel.h>

#include "config.cuh"

extern __device__ void mapper(curandState *state, value_type *value);

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}


__global__ void setup_kernel(curandState *state) {
    size_t id = threadIdx.x + blockIdx.x * blockDim.x;
    /* Each thread gets same seed, a different sequence
       number, no offset */
    curand_init(1234, id, 0, &state[id]);
}


/*
    Mapping Kernel: Since each mapper runs independently of each other, we can
    give each thread its own input to process and a disjoint space where it can`
    store the key/value pairs it produces.
*/
__global__ void mapKernel(curandState *state, value_type *values) {
    size_t threadId = blockIdx.x * blockDim.x + threadIdx.x;    // Global id of the thread
    // Total number of threads, by jumping this much, it ensures that no thread gets the same data
    size_t jump = blockDim.x * gridDim.x;

    // curand_init(1234, threadId, 0, &state[threadId]);

    curandState localState = state[threadId];

    for (size_t i=threadId; i<NUM_INPUT; i+=jump) {
        // Input data to run mapper on, and the location to place the output
        mapper(&localState, &values[i]);
    }
}

/*
    Call Mapper kernel with the required grid, blocks
    TODO: Err checking
*/
void runMapper(curandState *state, value_type *dev_values) {
    setup_kernel<<<GRID_SIZE, BLOCK_SIZE>>>(state);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    mapKernel<<<GRID_SIZE, BLOCK_SIZE>>>(state, dev_values);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
}


/*
    Main function to run Map-Reduce program
*/
void runMapReduce(output_type *output) {
    // 1. Allocate memory on GPU for Value array
    // 2. Initialize cuRAND
    // 3. Run Mapper kernel, which calls mapper function for the inputs decided for that thread
    // 4. Reduce using thrust to find total points
    // Calculate Pi and assign to output. Done!

    curandState *dev_states;

    // Pointers for input & value arrays
    value_type *dev_values;


    const uint64_cu TOTAL_THREADS = GRID_SIZE * BLOCK_SIZE;
    gpuErrchk( cudaMalloc((void **)&dev_states, TOTAL_THREADS * sizeof(curandState)) );

    // Allocate memory for value array
    size_t value_size = NUM_INPUT * sizeof(value_type);
    cudaMalloc(&dev_values, value_size);

    // Run mapper
    // This will run mapper kernel on all the inputs, and produces the key-value pairs
    runMapper(dev_states, dev_values);

    // Sum up the array using thrust::reduce
    thrust::device_ptr<value_type> dev_value_thrust_ptr = thrust::device_pointer_cast(dev_values);
    uint64_cu total_points = thrust::count(thrust::device, dev_value_thrust_ptr, dev_value_thrust_ptr + NUM_INPUT, true);

    // std::cout << "Total points: " << total_points << std::endl;
    *output = 4.0 * (double(total_points) / NUM_INPUT);

    // Free all memory
    cudaFree(dev_values);
}
