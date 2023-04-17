#include <Eigen/Core>
#include <iostream>
#include <fstream>
#include <random>
#include <chrono>

using Eigen::MatrixXd;
using Eigen::VectorXd;

#include <numeric>

// ...

MatrixXd systematic_resample(const MatrixXd& particles, const VectorXd& weights, std::mt19937& rng) {
    int num_particles = particles.rows();
    MatrixXd resampled_particles(num_particles, particles.cols());
    double step_size = 1.0 / num_particles;
    double r = std::uniform_real_distribution<double>(0.0, step_size)(rng);

    // Compute partial sums of the weights
    std::vector<double> partial_sums(num_particles);
    std::partial_sum(weights.data(), weights.data() + num_particles, partial_sums.begin());

    // Resample
    int i = 0;
    for (int m = 0; m < num_particles; m++) {
        while (r > partial_sums[i]) {
            i++;
        }
        resampled_particles.row(m) = particles.block(i, 0, 1, particles.cols());
        r += step_size;
        if (r >= 1.0) {
            r -= 1.0;
            i = 0;
        }
    }

    return resampled_particles;
}




int main() {
    // Set up the particle dimensions
    const int MIN_NUM_PARTICLES = 1;
    const int MAX_NUM_PARTICLES = 10000000;
    const int NUM_PARTICLES_STEP = 10;

    // Generate some sample data
    int particle_dimension = 10;
    // VectorXd weights = VectorXd::Random(MAX_NUM_PARTICLES);

    // Open the output file
    std::ofstream output_file;
    output_file.open("execution_times_systematic.csv");
    std::mt19937 rng;
    int iterations = 10;
    double elapsed_time = 0;

    // Loop over the different numbers of particles and time the C++ version of the function
    for (int num_particles = MIN_NUM_PARTICLES; num_particles <= MAX_NUM_PARTICLES; num_particles *= NUM_PARTICLES_STEP) {
        elapsed_time = 0;
        std::cout << "Number of particles: " << num_particles << "\n";
        for (int num_iterations = 0; num_iterations < iterations; num_iterations++) {
            std::cout <<"\tIteration number: " << num_iterations << "\n";
            MatrixXd particles = MatrixXd::Random(num_particles, particle_dimension);
            VectorXd weights = VectorXd::Ones(num_particles) / num_particles;
            auto start_time = std::chrono::high_resolution_clock::now();
            MatrixXd resampled_particles = systematic_resample(particles, weights.head(num_particles),rng);
            auto end_time = std::chrono::high_resolution_clock::now();
            elapsed_time += std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count() / 1e6;
        }

        // Write the execution time and number of particles to the output file
        output_file << num_particles << "," << elapsed_time/iterations << std::endl;
    }

    // Close the output file
    output_file.close();

    return 0;
}