#ifndef MAP_REDUCE_CUH
#define MAP_REDUCE_CUH

// GPU parameters
const int GRID_SIZE = 5;
const int BLOCK_SIZE = 1024;

using uint64_cu = unsigned long long int;

// No. of input elements (Length of dataset)
// TODO: Maybe make it variable, calculated from reading the text file
const uint64_cu NUM_INPUT = 5000;
// No. of pairs per input element
const int NUM_PAIRS = 1;
// Total No. of output values (K - No. of clusters)
const uint64_cu NUM_OUTPUT = 15;

// No. of values in each line (Size of datapoint)
const int LENGTH = 2;
// No. of iterations
const int ITERATIONS = 300;

// Custom types
struct Point {
    int values[LENGTH];
};


// Type declarations for input, output & key-value pairs
using input_type = Point;       // Datapoint (or vector) read from the text file
using output_type = Point;     // Outputs are the cluster centroids

// So each point will get associated with a cluster (with id -> key)
using key_type = int;               // Cluster that the point corresponds to
using value_type = Point;           // Point associated with the cluster


// Pair type definition
struct pair_type {
    key_type key;
    value_type value;

    // Printing for debugging
    friend std::ostream& operator<<(std::ostream& os, const pair_type& pair) {
        os << "Key: " << pair.key << ", Point: ";
        for (int i=0; i<LENGTH; i++)
            os << pair.value.values[i] << " ";

        os << "\n";
        return os;
    }
};

/*
    Comparision operator for comparing between 2 KeyValuePairs
    Returns true if first pair has key less than the second
*/
struct KeyValueCompare {
    __host__ __device__ bool operator()(const pair_type& lhs, const pair_type& rhs) {
        return lhs.key < rhs.key;
    }
};


const uint64_cu TOTAL_PAIRS = NUM_INPUT * NUM_PAIRS;

void runMapReduce(const input_type* input, output_type *output);

#endif // MAP_REDUCE_CUH
