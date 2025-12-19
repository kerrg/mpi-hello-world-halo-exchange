# Compiler and Runner
CXX = mpicxx
RUN = mpirun
CXXFLAGS = -Wall -O2 -std=c++11

# Project files
TARGET = bin/mpi_stencil
SRC = mpi_stencil.cc HaloExchangeUtils.cc
OBJ = $(addprefix bin/,$(SRC:.cc=.o))

# Default target: just build
all: $(TARGET)

# Build and run the executable
run: all
	@echo "--- Running $(TARGET) with 2 processes ---"
	$(RUN) -n 2 $(TARGET)

# Link the executable
$(TARGET): $(OBJ)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(OBJ)

# Compile object files
bin/%.o: %.cc
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Clean up build artifacts and log files
clean:
	rm -f $(TARGET) $(OBJ) bin/log_rank_*.txt
	rmdir -p bin 2>/dev/null || true # Remove bin directory if empty
	@echo "Cleaned up executable, object files, and log files."

.PHONY: all clean run