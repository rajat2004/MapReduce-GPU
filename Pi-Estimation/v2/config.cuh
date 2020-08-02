#ifndef MAP_REDUCE_CUH
#define MAP_REDUCE_CUH

// GPU parameters
const int GRID_SIZE = 1024;
const int BLOCK_SIZE = 1024;

// Input type
struct Point {
    double x;
    double y;
};

using uint64_cu = unsigned long long int;

// No. of input elements
const uint64_cu NUM_INPUT = 1e8;
// No. of pairs per input element
const int NUM_PAIRS = 1;
// No. of output values
const uint64_cu NUM_OUTPUT = 1;


// Type declarations for input, output & key-value pairs
using input_type = Point;       // Generated points
using key_type = char;          // We're using a single key for all pairs -> 0, so no need to use a bigger key
using value_type = bool;        // Whether point lies in the circle or outside, using `int` would just increase memory usage
using output_type = double;     // Output is estimated value of Pi


// Edit below this shouldn't be required

const uint64_cu TOTAL_PAIRS = NUM_INPUT * NUM_PAIRS;

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
        return lhs.key < rhs.key;
    }
};

void runMapReduce(input_type *input, output_type *output);

#endif // MAP_REDUCE_CUH
