#include <iostream>
#include <chrono>
#include <iomanip>
#include "random_generator.hpp"


const uint64_t NUM_SAMPLES = 1e9;
const int THREADS = 12;


uint64_t samplesInsideCircle(const uint64_t& num_samples) {
    UniformDistribution distribution;
    uint64_t circle_points = 0;

    for (uint64_t s=0; s!=num_samples; ++s) {
        auto x = distribution.sample();
        auto y = distribution.sample();

        if (x*x + y*y <= 1) {
            circle_points++;
        }
    }

    return circle_points;
}

double parallelPi() {
    const uint64_t samples_per_chunk = NUM_SAMPLES / THREADS;
    uint64_t circle_points = 0;

    // OpenMP parallelization with addition reduction
    #pragma omp parallel for reduction(+:circle_points)
    for (int i=0; i<THREADS; ++i) {
        circle_points += samplesInsideCircle(samples_per_chunk);
    }

    return 4.0f * (double(circle_points) / NUM_SAMPLES);
}


int main() {
    using millis = std::chrono::milliseconds;
    using std::chrono::duration_cast;
    using std::chrono::steady_clock;

    std::cout << "Parallel (OpenMP) version: \n";
    std::cout << "Number of samples: " << NUM_SAMPLES << " Threads: " << THREADS << std::endl;

    auto t_seq_1 = steady_clock::now();
    const auto approx_pi = parallelPi();
    auto t_seq_2 = steady_clock::now();

    std::cout << "Real Pi: 3.141592653589...\n";
    std::cout << "Approx Pi: " << std::setprecision(12) << approx_pi << std::endl;

    auto time1 = duration_cast<millis>( t_seq_2 - t_seq_1 ).count();
    std::cout << "Time: " << time1 << " milliseconds\n";
}
