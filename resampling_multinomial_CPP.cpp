#include <iostream>
#include <fstream>
#include <random>
#include <chrono>
#include <Eigen/Dense>

using Eigen::MatrixXd;
using Eigen::VectorXd;

MatrixXd multinomial_resampling_cpp(const MatrixXd &particles, const VectorXd &weights)
{
    int num_particles = particles.rows();
    int particle_dimension = particles.cols();

    // Normalize the weights
    double sum_weights = weights.sum();
    VectorXd normalized_weights = weights / sum_weights;

    // Resample particles using the multinomial distribution
    std::random_device rd;
    std::mt19937 gen(rd());
    std::discrete_distribution<int> d(normalized_weights.data(), normalized_weights.data() + normalized_weights.size());
    MatrixXd resampled_particles(num_particles, particle_dimension);
    for (int i = 0; i < num_particles; i++)
    {
        int index = d(gen);
        resampled_particles.row(i) = particles.row(index);
    }

    return resampled_particles;
}

int main()
{
    // Set up the particle dimensions
    const int MIN_NUM_PARTICLES = 1;
    const int MAX_NUM_PARTICLES = 10000000;
    const int NUM_PARTICLES_STEP = 10;

    // Generate some sample data
    int particle_dimension = 1;
    VectorXd weights = VectorXd::Random(MAX_NUM_PARTICLES);

    // Open the output file
    std::ofstream output_file;
    output_file.open("execution_times_multinomial.csv");
    int iterations = 10;
    double elapsed_time = 0;

    // Loop over the different numbers of particles and time the C++ version of the function
    for (int num_particles = MIN_NUM_PARTICLES; num_particles <= MAX_NUM_PARTICLES; num_particles *= NUM_PARTICLES_STEP)
    {
        elapsed_time = 0;
        std::cout << "Number of particles: " << num_particles << "\n";
        for (int num_iterations = 0; num_iterations < iterations; num_iterations++)
        {
            std::cout << "\tIteration number: " << num_iterations << "\n";
            MatrixXd particles = MatrixXd::Random(num_particles, particle_dimension);

            auto start_time = std::chrono::high_resolution_clock::now();
            MatrixXd resampled_particles = multinomial_resampling_cpp(particles, weights.head(num_particles));
            auto end_time = std::chrono::high_resolution_clock::now();
            elapsed_time += std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count() / 1e6;
        }

        // Write the execution time and number of particles to the output file
        output_file << num_particles << "," << elapsed_time / iterations << std::endl;
    }

    // Close the output file
    output_file.close();

    return 0;
}