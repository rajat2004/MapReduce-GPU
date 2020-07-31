#include <random>
#include <chrono>
#include <stdio.h>

// All random generators are generating integers in range [0, RAND_MAX]

// C stdlib's random number generator
class CRand {
public:
    CRand() {
        printf("Using C's rand() for random generator\n");
        int seed = std::chrono::system_clock::now().time_since_epoch().count();
        srand(seed);
    }

    int sample() {
        return rand();
    }
};

// C++11 Uniform Random Number Generator
class UniformDistribution {
public:
    UniformDistribution()
        : generator(std::chrono::system_clock::now().time_since_epoch().count()),
          distribution(0, RAND_MAX)
    {
        printf("Using C++11's uniform_int_distribution for random generator\n");
    }

    int sample() {
        return distribution(generator);
    }

private:
    std::default_random_engine generator;
    std::uniform_int_distribution<int> distribution;
};


// C++11 Normal Distribution
// Well, not exactly normal since values not in [0, RAND_MAX] are discarded, but close
class NormalDistribution {
public:
    NormalDistribution()
        : generator(std::chrono::system_clock::now().time_since_epoch().count()),
          distribution(mean, deviation)
    {
        printf("Using C++11's normal_distribution with Mean %f & Deviation %f\n",
                mean, deviation);
    }

    int sample() {
        // Discard values which don't lie between [0, RAND_MAX]
        int value;
        do {
            value = distribution(generator);
        } while (value<0 || value>RAND_MAX);

        return value;
    }

private:
    std::default_random_engine generator;
    const float mean = RAND_MAX/2.0;
    const float deviation = RAND_MAX/10.0;
    std::normal_distribution<float> distribution;
};


// C++11 Poisson Distribution
class PoissonDistribution {
public:
    PoissonDistribution()
        : generator(std::chrono::system_clock::now().time_since_epoch().count()),
          distribution(mean)
    {
        printf("Using C++11's poisson_distribution with Mean %d\n", mean);
    }

    int sample() {
        return distribution(generator);
    }

private:
    std::default_random_engine generator;
    const int mean = RAND_MAX/2;
    std::poisson_distribution<int> distribution;
};
