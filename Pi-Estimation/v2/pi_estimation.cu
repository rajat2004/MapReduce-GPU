#include <iostream>
#include <chrono>
#include "config.cuh"
#include "random_generator.hpp"


/*
    Mapper function for each input element
    Input is already stored in memory, and output pairs must be stored in the memory allocated
    Muliple pairs can be generated for a single input, but their number shouldn't exceed NUM_PAIRS
*/
__device__ void mapper(input_type *input, pair_type *pairs) {
    // Set key of the pair to 0
    // Single key for all the inputs
    pairs->key = 0;

    auto x = input->x;
    auto y = input->y;

    if (x*x + y*y <= 1)
        pairs->value = true;    // Point lies inside circle
    else
        pairs->value = false;   // Outside circle
}

/*
    Reducer to convert Key-Value pairs to desired output
    `len` number of pairs can be read starting from pairs, and output is stored in memory
*/
__device__ void reducer(pair_type *pairs, size_t len, output_type *output) {
    // Count number of points within unit circle
    int circle_points = 0;
    // TODO: Since this is just a reduction over the array, maybe replace with a thrust library function?
    for (pair_type* pair=pairs; pair!=(pairs+len); ++pair) {
        // value is a bool, 1 if point inside circle, else 0
        circle_points += pair->value;
    }

    *output = 4.0f * (double(circle_points) / len);
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
