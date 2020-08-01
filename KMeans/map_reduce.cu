#include <iostream>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>

#include "config.cuh"

extern __device__ void mapper(const input_type* input, pair_type *pairs, output_type *output);
extern __device__ void reducer(pair_type *pairs, size_t len, output_type *output);


/*
    Mapping Kernel: Since each mapper runs independently of each other, we can
    give each thread its own input to process and a disjoint space where it can`
    store the key/value pairs it produces.
*/
__global__ void mapKernel(const input_type* input, pair_type *pairs, output_type *dev_output) {
    size_t threadId = blockIdx.x * blockDim.x + threadIdx.x;    // Global id of the thread
    // Total number of threads, by jumping this much, it ensures that no thread gets the same data
    size_t jump = blockDim.x * gridDim.x;

    for (size_t i=threadId; i<NUM_INPUT; i+=jump) {
        // Input data to run mapper on, and the starting index of memory assigned for key-value pairs for this
        mapper(&input[i], &pairs[i * NUM_PAIRS], dev_output);
    }
}

/*
    Call Mapper kernel with the required grid, blocks
    TODO: Err checking
*/
void runMapper(const input_type* dev_input, pair_type *dev_pairs, output_type *dev_output) {
    mapKernel<<<GRID_SIZE, BLOCK_SIZE>>>(dev_input, dev_pairs, dev_output);
}


/*
    Reducer kernel
    Input is sorted array of keys (well, pairs)
    For each thread, find the keys that it'll work on and the range associated with each key
*/
__global__ void reducerKernel(pair_type *pairs, output_type *output) {
    size_t threadId = blockIdx.x * blockDim.x + threadIdx.x;    // Global id of the thread
    // Total number of threads, by jumping this much, it ensures that no thread gets the same data
    size_t jump = blockDim.x * gridDim.x;
    // printf("TID: %d, Jmp: %llu\n", threadId, jump);

    for (size_t i=threadId; i<NUM_OUTPUT; i+=jump) {
        // So now i is like the threadId that we need to run on
        // For each threadId, find the key associated with it (starting index, and the number of pairs)
        // And handle the case when there's no such key (no. of keys < no. of threads)
        size_t start_index = 0;                // Starting index of the key in the array of pairs
        size_t end_index = TOTAL_PAIRS;        // Ending index of the key in array of pairs
        size_t uniq_key_index = 0;                // In a list of unique sorted keys, the index of the key
        size_t value_size = 0;                 // No. of pairs for this key

        // TODO: Can this be converted to a single pass over the entire array once?
        // Before the reducer
        // Store unique keys and their ranges
        for (size_t j=1; j<TOTAL_PAIRS; j++) {
            if (KeyValueCompare()(pairs[j-1], pairs[j])) {
                // The keys are unequal, therefore we have moved on to a new key
                if (uniq_key_index == i) {
                    // The previous key was the one associated with this thread
                    // And we have reached the end of pairs for that key
                    // So we now know the start and end for the key, no need to go through more pairs
                    end_index = j;
                    break;
                }
                else {
                    // Still haven't reached the key required
                    // Increae the uniq_key_index since it's a new key, and store its starting index
                    uniq_key_index++;
                    start_index = j;
                }
            }
        }

        // printf("TID: %d, Uniq_key_index: %llu start_index: %llu, end_index: %llu\n",
        //         threadId, uniq_key_index, start_index, end_index);

        // We can have that the thread doesn't need to process any key
        if (uniq_key_index != i) {
            return;             // Enjoy, nothing to be done!
        }

        // Total number of pairs to be processed is end-start
        value_size = end_index - start_index;

        // Run the reducer
        reducer(&pairs[start_index], value_size, &output[i]);
    }
}

/*
    Call Reducer kernel with required grid, blocks
    TODO: Err checking
    TODO: Add separate constants for mapper, reducer grid, blocks
*/
void runReducer(pair_type *dev_pairs, output_type *dev_output) {
    reducerKernel<<<GRID_SIZE, BLOCK_SIZE>>>(dev_pairs, dev_output);
}


/*
    Main function to run Map-Reduce program
*/
void runMapReduce(const input_type* input, output_type *output) {
    // 1. Allocate memory on GPU for inputs, key-value pairs & centroids on GPU
    //      Note output memory is also allocated right now itself
    // 2. Copy inputs, initial centroids to GPU
    // 3. For the required iterations, do the following -
    //      1. Run Mapper kernel, which calls mapper function for the inputs decided for that thread
    //      2. Sort Key-Value pairs
    //      3. Reducer: Each thread gets a specific cluster, and finds the new centroids
    // 4. Copy output from GPU to host memory
    // 5. Free all GPU memory
    // Done! Finally

    // Pointers for input, key-value pairs & output on device
    input_type *dev_input;
    output_type *dev_output;
    pair_type *dev_pairs;

    // Allocate memory on GPU for input
    size_t input_size = NUM_INPUT * sizeof(input_type);
    cudaMalloc(&dev_input, input_size);

    // Allocate memory for key-value pairs
    size_t pair_size = TOTAL_PAIRS * sizeof(pair_type);
    cudaMalloc(&dev_pairs, pair_size);

    // Allocate memory for outputs
    // Since centroids are needed in K Means the entire time
    size_t output_size = NUM_OUTPUT * sizeof(output_type);
    cudaMalloc(&dev_output, output_size);


    // Allocate memory on host for older centroids
    // output_type *old_output = (output_type *) malloc(output_size);


    // Copy input datapoints to device
    cudaMemcpy(dev_input, input, input_size, cudaMemcpyHostToDevice);

    // Copy initial centroids to device
    cudaMemcpy(dev_output, output, output_size, cudaMemcpyHostToDevice);

    // Now run K Means for the specified iterations
    for (int iter=0; iter<ITERATIONS; iter++) {
        // printf("Starting iteration %d\n", iter);

        // Copy older centroids
        // cudaMemcpy(old_output, dev_output, output_size, cudaMemcpyDeviceToHost);

        // Run mapper
        // This will run mapper kernel on all the inputs, and produces the key-value pairs
        // It also requires the current centroids, so pass `dev_output` as well
        runMapper(dev_input, dev_pairs, dev_output);

        // Create Thrust device pointer from key-value pairs for sorting
        thrust::device_ptr<pair_type> dev_pair_thrust_ptr(dev_pairs);

        // thrust::copy(dev_pair_thrust_ptr, dev_pair_thrust_ptr + TOTAL_PAIRS, std::ostream_iterator<pair_type>(std::cout, " "));

        // Sort Key-Value pairs based on Key
        // This should run on the device itself
        thrust::sort(dev_pair_thrust_ptr, dev_pair_thrust_ptr + TOTAL_PAIRS, KeyValueCompare());

        // thrust::copy(dev_pair_thrust_ptr, dev_pair_thrust_ptr + TOTAL_PAIRS, std::ostream_iterator<pair_type>(std::cout, " "));

        // Run reducer kernel on key-value pairs
        runReducer(dev_pairs, dev_output);

        cudaDeviceSynchronize();

        // Copy new centroids
        // cudaMemcpy(output, dev_output, output_size, cudaMemcpyDeviceToHost);


    }

    // Copy outputs from GPU to host
    // Note host memory has already been allocated
    cudaMemcpy(output, dev_output, output_size, cudaMemcpyDeviceToHost);

    // Free all memory allocated on GPU
    cudaFree(dev_input);
    cudaFree(dev_pairs);
    cudaFree(dev_output);
}
