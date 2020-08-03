#include <iostream>
#include <chrono>
#include <fstream>
#include <sstream>
#include <string>
#include "config.cuh"
#include "random_generator.hpp"

const bool SAVE_TO_FILE = true;

__device__ __host__
uint64_cu distance(const Point& p1, const Point& p2) {
    uint64_cu dist = 0;
    for (int i=0; i<DIMENSION; i++) {
        int temp = p1.values[i]-p2.values[i];
        dist += temp * temp;
    }

    return dist;
}

/*
    Mapper function for each input element
    Input is already stored in memory, and output pairs must be stored in the memory allocated
    Muliple pairs can be generated for a single input, but their number shouldn't exceed NUM_PAIRS
*/
__device__ void mapper(const input_type* input, pair_type *pairs, output_type *output) {
    // Find centroid with min distance from the current point
    uint64_cu min_distance=ULLONG_MAX;
    int cluster_id=-1;

    for (int i=0; i<NUM_OUTPUT; i++) {
        uint64_cu dist = distance(*input, output[i]);
        if (dist < min_distance) {
            min_distance = dist;
            cluster_id = i;
        }
    }

    pairs->key = cluster_id;
    pairs->value = *input;
}


/*
    Reducer to convert Key-Value pairs to desired output
    `len` number of pairs can be read starting from pairs, and output is stored in memory
*/
__device__ void reducer(pair_type *pairs, size_t len, output_type *output) {
    // printf("Key: %d, Length: %llu\n", pairs[0].key, len);

    // Find new centroid
    uint64_cu new_values[DIMENSION];    // uint64_cu to avoid overflow
    for (int i=0; i<DIMENSION; i++)
        new_values[i] = 0;

    for (size_t i=0; i<len; i++) {
        for (int j=0; j<DIMENSION; j++)
            new_values[j] += pairs[i].value.values[j];    // Wow, this is bad naming
    }

    // uint64_cu diff = 0;

    // Take the key of any pair
    int cluster_idx = pairs[0].key;
    for (int i=0; i<DIMENSION; i++) {
        new_values[i]/=len;

        // diff += abs((int)new_values[i] - output[cluster_idx].values[i]);
        output[cluster_idx].values[i] = new_values[i];
    }

    // printf("Key: %d, Diff: %llu\n", cluster_idx, diff);
}



/*
    Initialize according to normal KMeans
    Choose K random data points as initial centroids
*/
void initialize(input_type *input, output_type *output) {
    // Uniform Number generator for random datapoints
    UniformDistribution distribution(NUM_INPUT);

    // Now chose initial centroids
    for (int i=0; i<NUM_OUTPUT; i++) {
        int sample = distribution.sample();
        output[i] = input[sample];
    }
}


/*
    KMeans++ initializer, takes longer
*/
void pp_initialize(input_type *input, output_type *output) {
    // Uniform Number generator for the first random chosen centroid
    UniformDistribution distribution(NUM_INPUT);

    int sample = distribution.sample();
    output[0] = input[sample];

    // Chose the next k-1 centroids
    for (int cluster_id=1; cluster_id<NUM_OUTPUT; cluster_id++) {
        uint64_cu max_dist_allp = 0;    // For storing max dist till now
        int max_dist_idx = -1;          // Index of datapoint at max distance

        for (int i=0; i<NUM_INPUT; i++) {
            // Find min dist between this point and all the prev centroids
            uint64_cu min_dist=ULLONG_MAX;
            for (int j=0; j<cluster_id; j++) {
                min_dist = min(min_dist, distance(input[i], output[j]));
            }

            // If this point is at max dist from all centroids
            if (min_dist > max_dist_allp) {
                max_dist_allp = min_dist;
                max_dist_idx = i;
            }
        }

        // Assign new centroid
        output[cluster_id] = input[max_dist_idx];
    }
}


/*
    Main function that runs a map reduce job.
*/
int main(int argc, char *argv[]) {
    using millis = std::chrono::milliseconds;
    using std::chrono::duration_cast;
    using std::chrono::steady_clock;
    using std::string;

    if (argc!=2) {
        printf("Requires 1 argument, name of input textfile\n");
        exit(1);
    }

    string filename = argv[1];

    auto t_seq_1 = steady_clock::now();

    // Allocate host memory
    size_t input_size = NUM_INPUT * sizeof(input_type);
    input_type *input = (input_type *) malloc(input_size);

    size_t output_size = NUM_OUTPUT * sizeof(output_type);
    output_type *output = (output_type *) malloc(output_size);

    // Read data from text file
    string line;
    std::ifstream input_file(filename);

    if (input_file.is_open()) {
        for (size_t line_idx=0; line_idx<NUM_INPUT; line_idx++) {
            getline(input_file, line);
            std::istringstream buffer(line);

            for (int i=0; i<DIMENSION; i++)
                buffer >> input[line_idx].values[i];
        }

        input_file.close();
    }
    else {
        std::cout << "Error while opening file: " << filename;
        exit(1);
    }

    // Now chose initial centroids
    initialize(input, output);
    // pp_initialize(input, output);

    auto t_seq_2 = steady_clock::now();

    // Run the Map Reduce Job
    runMapReduce(input, output);

    // Save output if required
    std::ofstream output_file;
    if (SAVE_TO_FILE) {
        string output_filename = filename + ".output";
        output_file.open(output_filename);
        if (!output_file.is_open()) {
            std::cout << "Unable to open output file: " << output_filename;
            exit(1);
        }
    }

    printf("Centroids: \n");
    // Iterate through the output array
    for (size_t i=0; i<NUM_OUTPUT; i++) {
        for (int j=0; j<DIMENSION; j++) {
            printf("%d ", output[i].values[j]);
            if (SAVE_TO_FILE)
                output_file << output[i].values[j] << " ";
        }

        printf("\n");
        if (SAVE_TO_FILE)
            output_file << "\n";
    }

    // Free host memory
    free(input);
    free(output);

    auto t_seq_3 = steady_clock::now();

    auto time1 = duration_cast<millis>( t_seq_2 - t_seq_1 ).count();
    auto time2 = duration_cast<millis>( t_seq_3 - t_seq_2 ).count();
    auto total_time = duration_cast<millis>( t_seq_3 - t_seq_1 ).count();

    std::cout << "Time for CPU data loading + initialize: " << time1 << " milliseconds\n";
    std::cout << "Time for map reduce KMeans + writing outputs + free: " << time2 << " milliseconds\n";
    std::cout << "Total time: " << total_time << " milliseconds\n";

    return 0;
}
