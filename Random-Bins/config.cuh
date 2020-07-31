#ifndef MAP_REDUCE_CUH
#define MAP_REDUCE_CUH

// GPU parameters
const int GRID_SIZE = 1024;
const int BLOCK_SIZE = 1024;

using uint64_cu = unsigned long long int;

// No. of input elements
const uint64_cu NUM_INPUT = 1e8;
// No. of pairs per input element
const int NUM_PAIRS = 1;
// Total No. of output values (10 Bins)
const uint64_cu NUM_OUTPUT = 10;

// Custom types
struct Bin {
    int start;
    int end;
};

struct BinFrequency {
    Bin bin;
    int count;
};

// Size of each bin
const int BIN_SIZE = RAND_MAX / NUM_OUTPUT;

// Type declarations for input, output & key-value pairs
using input_type = int;               // Generated random numbers (only taking till RAND_MAX, so int is okay)
using key_type = Bin;                 // Bin corresponding to the value
using value_type = char;              // This is not actually used, so go for the lowest one
using output_type = BinFrequency;     // Outputs are the different bins with their counts

// Pair type definition
struct pair_type {
    key_type key;
    value_type value;
};

/*
    Comparision operator for comparing between 2 KeyValuePairs
    Returns true if first pair has key less than the second
*/
struct KeyValueCompare {
    __host__ __device__ bool operator()(const pair_type& lhs, const pair_type& rhs) {
        return lhs.key.start < rhs.key.start;
    }
};


const uint64_cu TOTAL_PAIRS = NUM_INPUT * NUM_PAIRS;

void runMapReduce(input_type *input, output_type *output);

#endif // MAP_REDUCE_CUH
