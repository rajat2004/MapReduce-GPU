#include <random>
#include <chrono>

// For generating random numbers between -1,1
class UniformDistribution
{
public:
    UniformDistribution()
        : generator(), distribution(-1.0, 1.0)
    {
        int seed = std::chrono::system_clock::now().time_since_epoch().count();
        generator.seed(seed);
    }

    double sample() {
        return distribution(generator);
    }

private:
    std::default_random_engine generator;
    std::uniform_real_distribution<double> distribution;
};
