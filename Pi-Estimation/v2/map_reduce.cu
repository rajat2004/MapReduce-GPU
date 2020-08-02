#include <iostream>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>

#include "config.cuh"

extern __device__ void mapper(const input_type* input, value_type *value);


/*
    Mapping Kernel: Since each mapper runs independently of each other, we can
    give each thread its own input to process and a disjoint space where it can`
    store the key/value pairs it produces.
*/
__global__ void mapKernel(const input_type* input, value_type *values) {
    size_t threadId = blockIdx.x * blockDim.x + threadIdx.x;    // Global id of the thread
    // Total number of threads, by jumping this much, it ensures that no thread gets the same data
    size_t jump = blockDim.x * gridDim.x;

    for (size_t i=threadId; i<NUM_INPUT; i+=jump) {
        // Input data to run mapper on, and the location to place the output
        mapper(&input[i], &values[i]);
    }
}

/*
    Call Mapper kernel with the required grid, blocks
    TODO: Err checking
*/
void runMapper(const input_type* dev_input, value_type *dev_values) {
    mapKernel<<<GRID_SIZE, BLOCK_SIZE>>>(dev_input, dev_values);
    cudaDeviceSynchronize();
}


/*
    Main function to run Map-Reduce program
*/
void runMapReduce(input_type *input, output_type *output) {
    // 1. Allocate memory on GPU for inputs
    // 2. Allocate momory for Value array
    // 3. Copy inputs to GPU
    // 4. Run Mapper kernel, which calls mapper function for the inputs decided for that thread
    // 5. Free input memory
    // 6. Reduce using thrust to find total points
    // Calculate Pi and assign to output. Done!

    // Pointers for input & value arrays
    input_type *dev_input;
    value_type *dev_values;

    // Allocate memory on GPU for input
    size_t input_size = NUM_INPUT * sizeof(input_type);
    cudaMalloc(&dev_input, input_size);

    // Allocate memory for value array
    size_t value_size = NUM_INPUT * sizeof(value_type);
    cudaMalloc(&dev_values, value_size);

    // Copy input data to device
    cudaMemcpy(dev_input, input, input_size, cudaMemcpyHostToDevice);

    // Run mapper
    // This will run mapper kernel on all the inputs, and produces the key-value pairs
    runMapper(dev_input, dev_values);

    // Sum up the array using thrust::reduce
    thrust::device_ptr<value_type> dev_value_thrust_ptr = thrust::device_pointer_cast(dev_values);
    uint64_cu total_points = thrust::count(thrust::device, dev_value_thrust_ptr, dev_value_thrust_ptr + NUM_INPUT, true);

    // std::cout << "Total points: " << total_points << std::endl;
    *output = 4.0 * (double(total_points) / NUM_INPUT);

    // Free all memory
    cudaFree(dev_input);
    cudaFree(dev_values);
}
