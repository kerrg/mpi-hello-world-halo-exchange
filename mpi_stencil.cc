#include <iostream>
#include <vector>
#include <fstream>
#include <utility> // For std::swap
#include <array>
#include <mpi.h>
#include "HaloExchangeUtils.h"

int main(int argc, char *argv[])
{
    using HaloExchangeUtils::kPointsPerRank;
    using HaloExchangeUtils::kBufferElemCount;
    using HaloExchangeUtils::kMpiRequestNum;
    using HaloExchangeUtils::kNumSteps;

    MPI_Init(&argc, &argv);

    int rank = 0;
    int world_size = 0;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    if (world_size < 2)
    {
        std::cerr << "This sample program should be run with a world size of at least "
                  << "2, instead found " << world_size << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    const int max_rank = world_size - 1;
    int left_neighbor = rank - 1;
    int right_neighbor = rank + 1;

    if (rank == 0)
    {
        left_neighbor = MPI_PROC_NULL;
    }

    if (rank == max_rank)
    {
        right_neighbor = MPI_PROC_NULL;
    }

    std::vector<double> current_data(kBufferElemCount);
    std::vector<double> next_data(kBufferElemCount);

    HaloExchangeUtils::initBuffer(rank, current_data, next_data);

    std::ofstream logfile = HaloExchangeUtils::initLogFile(rank);

    // Barrier call ensures all ranks start at the same time for a fair measurement
    MPI_Barrier(MPI_COMM_WORLD);
    double start_time = MPI_Wtime();

    std::array<MPI_Request, kMpiRequestNum> requests;
    for (int step = 0; step < kNumSteps; step++)
    {
        HaloExchangeUtils::doLogging(current_data, logfile, step);

        HaloExchangeUtils::initiateHaloExchange(current_data, kPointsPerRank, left_neighbor, right_neighbor, requests);

        // Now that the non-blocking send and receives have been setup, do the
        // interior work while we wait on the remote nodes to respond.
        for (size_t i = 2; i < kPointsPerRank; i++)
        {
            next_data[i] = (current_data[i - 1] + current_data[i] + current_data[i + 1]) / 3.0;
        }

        MPI_Waitall(kMpiRequestNum, requests.data(), MPI_STATUSES_IGNORE);

        // Now that we've received the halo values, or the edges from the remote
        // nodes, we can compute our own edges.
        next_data[1] = (current_data[0] + current_data[1] + current_data[2]) / 3.0;
        next_data[kPointsPerRank] =
            (current_data[kPointsPerRank - 1] + current_data[kPointsPerRank] + current_data[kPointsPerRank + 1]) /
            3.0;

        // Now copy next_data into current_data to prep for the next loop iteration.
        // To really optimize for speed, we could swap pointers here instead.
        std::swap(current_data, next_data);
    }

    double end_time = MPI_Wtime();
    double local_elapsed = end_time - start_time;

    double max_elapsed;
    MPI_Reduce(&local_elapsed, &max_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0)
    {
        std::cout << "\nSimulation took " << max_elapsed << " seconds using " << world_size << " ranks" << std::endl;
    }

    // --- Gather results --
    std::vector<double> global_results;
    if (rank == 0)
    {
        // Only Rank 0 needs to allocate memory for the total rod
        global_results.resize(world_size * kPointsPerRank);
    }

    // We gather starting from index 1 to skip the left halo
    MPI_Gather(&current_data[1], kPointsPerRank, MPI_DOUBLE, global_results.data(), kPointsPerRank, MPI_DOUBLE, 0,
               MPI_COMM_WORLD);
    if (rank == 0)
    {
        HaloExchangeUtils::printResults(world_size, max_elapsed, global_results);
    }

    MPI_Finalize();
    return 0;
}