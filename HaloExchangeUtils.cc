#include <iomanip>
#include <sstream>

#include "HaloExchangeUtils.h"

namespace HaloExchangeUtils {

void initiateHaloExchange(std::vector<double>& current_data,
                          int points_per_rank, int left_neighbor,
                          int right_neighbor,
                          std::array<MPI_Request, kMpiRequestNum>& requests) {
  CHECK(!current_data.empty());
  CHECK(requests.data());

  MPI_Irecv(&current_data[0], 1, MPI_DOUBLE, left_neighbor, kReservedMessageTag,
            MPI_COMM_WORLD, &requests[0]);

  MPI_Irecv(&current_data[points_per_rank + 1], 1, MPI_DOUBLE, right_neighbor,
            kReservedMessageTag, MPI_COMM_WORLD, &requests[1]);

  MPI_Isend(&current_data[1], 1, MPI_DOUBLE, left_neighbor, kReservedMessageTag,
            MPI_COMM_WORLD, &requests[2]);

  MPI_Isend(&current_data[points_per_rank], 1, MPI_DOUBLE, right_neighbor,
            kReservedMessageTag, MPI_COMM_WORLD, &requests[3]);
}

void logStepResults(const std::vector<double>& current_data,
                    std::ofstream& logfile, int step) {
  CHECK(logfile.is_open());

  if (kPointsPerRank <= 20) {
    logfile << "Step " << step << ": ";
    for (size_t j = 0; j < kBufferElemCount; j++) {
      logfile << std::fixed << std::setprecision(2);
      if (j == 0 || j == kPointsPerRank + 1) {
        logfile << "[" << std::setw(6) << current_data[j] << "] ";
      } else {
        logfile << std::setw(6) << current_data[j] << " ";
      }
    }
    logfile << std::endl;
  }
}

void initBuffer(int rank, std::vector<double>& current_data,
                std::vector<double>& next_data) {
  for (size_t i = 0; i < kBufferElemCount; i++) {
    if (rank == 0 && i > 0 && i <= kPointsPerRank) {
      current_data[i] = 100.0f;
    } else {
      current_data[i] = 0.0f;
    }
    next_data[i] = 0.0f;
  }
}

std::ofstream initLogFile(int rank) {
  std::ostringstream filename;
  filename << "bin/log_rank_" << rank << ".txt";
  std::ofstream logfile(filename.str());
  CHECK(logfile.is_open());
  return logfile;
}

void printResults(int world_size, double max_elapsed,
                  const std::vector<double>& global_results) {
  size_t total_points = world_size * kPointsPerRank;
  std::cout << "\n--- Simulation Summary (Total Points: " << total_points
            << ") ---" << std::endl;
  std::cout << "Execution Time: " << max_elapsed << " seconds" << std::endl;

  std::cout << std::fixed << std::setprecision(1);
  std::cout << "Samples: [Start: " << global_results[0]
            << "] ... [Mid: " << global_results[total_points / 2]
            << "] ... [End: " << global_results[total_points - 1] << "]"
            << std::endl;

  std::cout << "Heat Map: [";
  size_t num_chars = 40;
  for (size_t i = 0; i < num_chars; i++) {
    size_t idx = (i * total_points) / num_chars;
    double val = global_results[idx];
    if (val > 80) {
      std::cout << "H";  // Hot
    } else if (val > 40) {
      std::cout << "m";  // Medium
    } else if (val > 10) {
      std::cout << ".";  // Cooling
    } else {
      std::cout << "_";  // Cold
    }
  }
  std::cout << "]" << std::endl << std::endl;
}

}  // namespace HaloExchangeUtils
