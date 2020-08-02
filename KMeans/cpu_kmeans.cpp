#include <iostream>
#include <chrono>
#include <fstream>
#include <sstream>
#include <cstring>
#include <climits>
#include "random_generator.hpp"


// No. of input elements (Length of dataset)
// TODO: Maybe make it variable, calculated from reading the text file
const uint64_t NUM_INPUT = 5000;
// No. of values in each line (Size of datapoint)
const int DIMENSION = 2;
// No. of iterations
const int ITERATIONS = 1000;
// Total No. of output values (K - No. of clusters)
const int NUM_OUTPUT = 15;

// Custom types
struct Point {
    int values[DIMENSION];
};

// Type declarations for input, output & key-value pairs
using input_type = Point;       // Datapoint (or vector) read from the text file
using output_type = Point;      // Outputs are the cluster centroids

const bool SAVE_TO_FILE = true;


uint64_t distance(const Point& p1, const Point& p2) {
    uint64_t dist = 0;
    for (int i=0; i<DIMENSION; i++)
        dist += (p1.values[i]-p2.values[i]) * (p1.values[i]-p2.values[i]);

    return dist;
}


class Cluster {
private:
    int cluster_id;
    Point centroid;
    std::vector<Point> points;

public:
    Cluster(int id, const Point& init_centroid)
        : cluster_id(id), centroid(init_centroid)
    {}

    const Point& getCentroid() {
        return centroid;
    }

    void addPoint(const Point& point) {
        points.push_back(point);
    }

    void clear() {
        points.clear();
    }

    void updateCentroid() {
        uint64_t new_values[DIMENSION];
        memset(new_values, 0, DIMENSION*sizeof(uint64_t));

        for (const auto& p : points) {
            for (int i=0; i<DIMENSION; i++)
                new_values[i] += p.values[i];
        }

        for (int i=0; i<DIMENSION; i++)
            centroid.values[i] = new_values[i]/points.size();
    }
};


/*
    K Means on CPU
    TODO: OpenMP
*/
void runKMeans(const input_type* input, output_type *output) {
    // 1. Iterate over all datapoints, find nearest cluster
    // 2. Find new clusters

    // Initialize clusters with centroids
    std::vector<Cluster> clusters;
    for (int i=0; i<NUM_OUTPUT; i++) {
        clusters.push_back(Cluster(i, output[i]));
    }

    for (int iter=0; iter<ITERATIONS; iter++) {
        // std::cout << "Iteration: " << iter << "/" << ITERATIONS << std::endl;

        // Iterate over all datapoints
        for (uint64_t i=0; i<NUM_INPUT; i++) {
            uint64_t min_dist = ULLONG_MAX;
            int c_id = -1;

            for (int j=0; j<NUM_OUTPUT; j++) {
                uint64_t temp_dist = distance(input[i], clusters[j].getCentroid());
                if (temp_dist < min_dist) {
                    min_dist = temp_dist;
                    c_id = j;
                }
            }

            // Assign this point to the cluster
            clusters[c_id].addPoint(input[i]);
        }

        // All the points have been assigned
        // Now update the Centroid of each cluster
        for (auto cluster : clusters) {
            cluster.updateCentroid();
            cluster.clear();            // Old points are no longer needed, remove so that they don't affect the next iteration
        }
    }

    // Copy centroids back to the output space
    for (int i=0; i<NUM_OUTPUT; i++)
        output[i] = clusters[i].getCentroid();
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
        uint64_t max_dist_allp = 0;    // For storing max dist till now
        int max_dist_idx = -1;          // Index of datapoint at max distance

        for (int i=0; i<NUM_INPUT; i++) {
            // Find min dist between this point and all the prev centroids
            uint64_t min_dist=ULLONG_MAX;
            for (int j=0; j<cluster_id; j++) {
                min_dist = std::min(min_dist, distance(input[i], output[j]));
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



int main(int argc, char const *argv[]) {
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
    initialize(input, output);          // Normal KMeans initialize (random) or
    // pp_initialize(input, output);       // KMeans++

    auto t_seq_2 = steady_clock::now();

    // Run  KMeans
    runKMeans(input, output);

    // Save output if required
    std::ofstream output_file;
    if (SAVE_TO_FILE) {
        string output_filename = filename + ".output.cpu";
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
    std::cout << "Time for KMeans + writing output + free: " << time2 << " milliseconds\n";
    std::cout << "Total time: " << total_time << " milliseconds\n";
    return 0;
}
