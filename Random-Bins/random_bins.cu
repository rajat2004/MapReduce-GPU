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
    // Find the bin where the value lies
    int bin_start = *input - (*input % BIN_SIZE);

    pairs->key.start = bin_start;
    pairs->key.end = bin_start + BIN_SIZE;
}


/*
    Reducer to convert Key-Value pairs to desired output
    `len` number of pairs can be read starting from pairs, and output is stored in memory
*/
__device__ void reducer(pair_type *pairs, size_t len, output_type *output) {
    // Number of pairs with the same bin is the count of that bin
    output->bin = pairs->key;
    output->count = len;
}


/*
    Main function that runs a map reduce job.
*/
int main() {
    using millis = std::chrono::milliseconds;
    using std::chrono::duration_cast;
    using std::chrono::steady_clock;

    // Specfiy the distribution to be used
    UniformDistribution distribution;
    // CRand distribution;
    // NormalDistribution distribution;
    // PoissonDistribution distribution;

    auto t_seq_1 = steady_clock::now();

    // Allocate host memory
    size_t input_size = NUM_INPUT * sizeof(input_type);
    input_type *input = (input_type *) malloc(input_size);

    size_t output_size = NUM_OUTPUT * sizeof(output_type);
    output_type *output = (output_type *) malloc(output_size);

    // Populate the input array with random coordinates
    printf("Generating %llu Test Points\n", NUM_INPUT);
    for (size_t i = 0; i < NUM_INPUT; i++) {
        input[i] = distribution.sample();
        // printf("%d\n", input[i]);
    }

    auto t_seq_2 = steady_clock::now();

    // Run the Map Reduce Job
    runMapReduce(input, output);

    // Iterate through the output array
    for (size_t i = 0; i < NUM_OUTPUT; i++) {
        printf("Bin [%d, %d): \t Count %d\n",
            output[i].bin.start, output[i].bin.end, output[i].count);
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
