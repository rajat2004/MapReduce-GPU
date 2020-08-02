#include <iostream>
#include <chrono>
#include "config.cuh"
#include "random_generator.hpp"


/*
    Mapper function for each input element
    Input is already stored in memory, and output pairs must be stored in the memory allocated
    Muliple pairs can be generated for a single input, but their number shouldn't exceed NUM_PAIRS
*/
__device__ void mapper(const input_type* input, value_type *value) {
    auto x = input->x;
    auto y = input->y;

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

    // Allocate host memory
    size_t input_size = NUM_INPUT * sizeof(input_type);
    input_type *input = (input_type *) malloc(input_size);

    size_t output_size = NUM_OUTPUT * sizeof(output_type);
    output_type *output = (output_type *) malloc(output_size);

    // Populate the input array with random coordinates
    printf("Generating %llu Test Points\n", NUM_INPUT);
    for (size_t i = 0; i < NUM_INPUT; i++) {
        input[i].x = distribution.sample();
        input[i].y = distribution.sample();
    }

    auto t_seq_2 = steady_clock::now();

    // Run the Map Reduce Job
    runMapReduce(input, output);

    // Iterate through the output array
    for (size_t i = 0; i < NUM_OUTPUT; i++) {
        printf("Value of Pi: %.12f\n", output[i]);
    }

    // Free host memory
    free(input);
    free(output);

    auto t_seq_3 = steady_clock::now();

    auto time1 = duration_cast<millis>( t_seq_2 - t_seq_1 ).count();
    auto time2 = duration_cast<millis>( t_seq_3 - t_seq_2 ).count();
    auto total_time = duration_cast<millis>( t_seq_3 - t_seq_1 ).count();

    std::cout << "Time for CPU data gen: " << time1 << " milliseconds\n";
    std::cout << "Time for map reduce (+free): " << time2 << " milliseconds\n";
    std::cout << "Total time: " << total_time << " milliseconds\n";

    return 0;
}
