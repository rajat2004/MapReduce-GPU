#include <iostream>
#include <chrono>
#include <curand_kernel.h>
#include "config.cuh"
#include "random_generator.hpp"


/*
    Mapper function for each input element
    Input is already stored in memory, and output pairs must be stored in the memory allocated
    Muliple pairs can be generated for a single input, but their number shouldn't exceed NUM_PAIRS
*/
 __device__ void mapper(curandState *state, value_type *value) {
    float x = curand_uniform(state);
    float y = curand_uniform(state);

    if (x*x + y*y <= 1)
        *value = true;    // Point lies inside circle
    else
        *value = false;   // Outside circle
}

/*
    Main function that runs a map reduce job.
*/
int main() {
    using millis = std::chrono::milliseconds;
    using std::chrono::duration_cast;
    using std::chrono::steady_clock;

    UniformDistribution distribution;

    auto t_seq_1 = steady_clock::now();

    size_t output_size = NUM_OUTPUT * sizeof(output_type);
    output_type *output = (output_type *) malloc(output_size);

    // Run the Map Reduce Job
    runMapReduce(output);

    // Iterate through the output array
    for (size_t i = 0; i < NUM_OUTPUT; i++) {
        printf("Value of Pi: %.12f\n", output[i]);
    }

    // Free host memory
    free(output);

    auto t_seq_2 = steady_clock::now();

    auto time = duration_cast<millis>( t_seq_2 - t_seq_1 ).count();

    std::cout << "No. of Samples: " << NUM_INPUT << std::endl;
    std::cout << "Total time: " << time << " milliseconds\n";

    return 0;
}
