#include <random>
#include <chrono>
#include <stdio.h>

// C++11 Uniform Random Number Generator
class UniformDistribution {
public:
    UniformDistribution(int max)
        : generator(std::chrono::system_clock::now().time_since_epoch().count()),
          distribution(0, max)
    {
        // printf("Using C++11's uniform_int_distribution for random generator\n");
    }

    int sample() {
        return distribution(generator);
    }

private:
    std::default_random_engine generator;
    std::uniform_int_distribution<int> distribution;
};
