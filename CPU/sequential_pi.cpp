#include <iostream>
#include <chrono>
#include <iomanip>
#include "random_generator.h"

const uint64_t NUM_SAMPLES = 1e9;


double approximatePi(const uint64_t& num_samples) {
    UniformDistribution distribution;
    uint64_t circle_points = 0;

    for (uint64_t s=0; s!=num_samples; ++s) {
        auto x = distribution.sample();
        auto y = distribution.sample();

        if (x*x + y*y <= 1)
            circle_points++;
    }

    return 4.0f * (double(circle_points) / num_samples);
}

double sequentialPi() {
    return approximatePi(NUM_SAMPLES);
}


int main() {
    using millis = std::chrono::milliseconds;
    using std::chrono::duration_cast;
    using std::chrono::steady_clock;

    auto t_seq_1 = steady_clock::now();
    const auto approxPi = sequentialPi();
    auto t_seq_2 = steady_clock::now();

    std::cout << "Sequential version: \n";
    std::cout << "number of samples: " << NUM_SAMPLES << std::endl;
    std::cout << "real Pi: 3.141592653589...\n";
    std::cout << "approx Pi: " << std::setprecision(12) << approxPi << std::endl;

    auto time1 = duration_cast<millis>( t_seq_2 - t_seq_1 ).count();
    std::cout << "Time: " << time1 << " milliseconds\n" << std::endl;
}
