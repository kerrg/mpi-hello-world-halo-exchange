#ifndef HALOEXCHANGE_HALOEXCHANGEUTILS_H
#define HALOEXCHANGE_HALOEXCHANGEUTILS_H

#include <mpi.h>
#include <vector>
#include <fstream>
#include <iostream>
#include <array>

/**
 * @brief A macro to check for a condition and abort via MPI if the condition is false.
 * @param test The condition to check.
 */
#define CHECK(test)                                                            \
  if (!(test)) {                                                               \
    std::cerr << "Check failed: " << #test << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
    MPI_Abort(MPI_COMM_WORLD, 1);                                              \
  }

/**
 * @brief A collection of utility functions and constants for the Halo Exchange MPI simulation.
 */
namespace HaloExchangeUtils {

    /// The number of data points each MPI rank is responsible for.
    constexpr size_t kPointsPerRank = 10;
    /// Total elements in each rank's buffer, including 2 halo points (one on each side).
    constexpr size_t kBufferElemCount = (kPointsPerRank + 2);
    /// The number of non-blocking MPI requests for the halo exchange (2 sends, 2 receives).
    constexpr size_t kMpiRequestNum = 4;
    /// A reserved MPI message tag for this simulation.
    constexpr size_t kReservedMessageTag = 2450;
    /// The number of steps (iterations) the simulation will run.
    constexpr int kNumSteps = 50;

    /**
     * @brief Initiates the non-blocking send and receive operations for the halo exchange.
     * @param current_data The data buffer for the current rank, including halo regions.
     * @param points_per_rank The number of data points this rank is responsible for (excluding halos).
     * @param left_neighbor The MPI rank of the left neighbor.
     * @param right_neighbor The MPI rank of the right neighbor.
     * @param requests An std::array to store the MPI_Request objects for the non-blocking operations.
     */
    void initiateHaloExchange(std::vector<double>& current_data,
                              int points_per_rank,
                              int left_neighbor,
                              int right_neighbor,
                              std::array<MPI_Request, kMpiRequestNum>& requests);

    /**
     * @brief Writes the current state of a rank's data buffer to a log file for a given simulation step.
     * @param current_data The data buffer to log.
     * @param logfile The output file stream to write the log to.
     * @param step The current simulation step number.
     */
    void logStepResults(const std::vector<double>& current_data, std::ofstream& logfile, int step);

    /**
     * @brief Initializes the data buffers for the simulation.
     * For rank 0, the local data points are initialized to 100.0; for all other ranks, they are 0.0.
     * Halo regions and the 'next' data buffer are initialized to 0.0 for all ranks.
     * @param rank The MPI rank of the current process.
     * @param current_data The primary data buffer to initialize.
     * @param next_data The secondary data buffer to initialize.
     */
    void initBuffer(int rank, std::vector<double>& current_data, std::vector<double>& next_data);

    /**
     * @brief Creates and opens a log file specific to an MPI rank.
     * The file will be created in the 'bin' directory.
     * @param rank The MPI rank to create the log file for.
     * @return An std::ofstream object for the newly created log file.
     */
    std::ofstream initLogFile(int rank);

    /**
     * @brief Prints a final summary of the simulation results. Should only be called by the root rank.
     * @param world_size The total number of MPI ranks in the simulation.
     * @param max_elapsed The maximum elapsed time recorded across all ranks.
     * @param global_results A vector containing the gathered results from all ranks.
     */
    void printResults(int world_size, double max_elapsed, const std::vector<double>& global_results);

} // namespace HaloExchangeUtils

#endif //HALOEXCHANGE_HALOEXCHANGEUTILS_H