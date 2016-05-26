# NearptD

NearptD is an exact nearest neighbor search program that utilizes a uniform grid to perform fast queries. It uses the GPU to parallelize both preprocessing and querying.

## Requirements

 * C++ compiler, edit makefile to use your compiler (default clang++3.5)
 * Nvidia CUDA, edit makefile to point to path (default /usr/local/cuda)
 * Thrust, should be packaged with CUDA
 
## Build

The makefile should handle building, there are a few different ways to compile, though. Simply run `make all` or whatever version you wish to create an executable. Because of the aggressive templating used in NearptD, the compilation is a single command, and should take around 45 seconds to a minute for up to 5 dimensions, but compile times increase greatly after that.

 * all: Default output, displays useful timing information after the program finishes.
 * debug: Useful for debugging if things aren't working, be warned this will print the entire input array, only use with small test input
 * profile: Displays more in-depth timing information as NearptD runs
 * exhaustive: The result of each query is checked against a result found via brute-force searching, to validate results.
 * stats: Displays statistics about the run, like a histogram of the number of points in each cell, and the types of queries run.
 * timing: Has no output except a single tab separated line of times, used when multiple runs are done in a script to collect timing information.
 * clean: Remove executable.
 
## Running

Regardless of how it is compiled, NearptD requires 4 command line arguments:

`./main ng_factor /path/to/fixed /path/to/queries /path/to/results`

*ng_factor* is the scaling factor of the uniform grid, usually in the range of 0.5 to 3, when in doubt use 1. The included main expects all point files to be in binary format, with each point being encoded as unsigned short ints. This means for 3 dimensional data, each point should be 6 bytes. Results will be the pair found for each query, encoded as the coordinates of the query point, then the coordinates of the closest fixed point found. This should be exactly twice as large as the query file. If you wish to read data in a different format, you can modify main to read points differently, though a warning that non-binary format will lead to I/O times that are likely longer than it will take NearptD to run.

If the GPU used to run NearptD also has an X11 service on it, the X11 service may become unresponsive if a large dataset is run. If NearptD requires too long to run, Thrust may timeout, to allow the X11 service to become responsive again. This is not a fault of NearptD, and can be fixed by running NearptD on a GPU without an X11 service running, either by using a different GPU or removing the X11 service.

NearptD can reach the limits of GPU memory on larger datasets. On a machine with 6GB of GPU memory, the largest dataset that was preprocessed and queried against was ~184 million points, though your mileage may vary.
