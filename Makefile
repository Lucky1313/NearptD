BIN := 1dtest

CC=gcc
CXX=g++
RM=rm -f

CUDA_INSTALL_PATH=/usr/local/cuda
CUDA=$(CUDA_INSTALL_PATH)/bin/nvcc
CUDA_LIBS=-L"$(CUDA_INSTALL_PATH)/lib64"

CXXFLAGS=-g -O3 -Wall
CUDAFLAGS=-O3 -arch=compute_35
DEBUGFLAGS=$(CUDAFLAGS) -DTHRUST_DEBUG -D=DEBUG

INCLUDE= -I"$(CUDA_INSTALL_PATH)/include"
SRC_DIR=src

all: 
	$(CUDA) $(CUDAFLAGS) $(INCLUDE) $(CUDA_LIBS) $(SRC_DIR)/main.cu -o main

debug:
	$(CUDA) $(DEBUGFLAGS) $(INCLUDE) $(CUDA_LIBS) $(SRC_DIR)/main.cu -o main

exhaustive:
	$(CUDA) $(CUDAFLAGS) -D=EXHAUSTIVE $(INCLUDE) $(CUDA_LIBS) $(SRC_DIR)/main.cu -o main

stats:
	$(CUDA) $(CUDAFLAGS) -D=STATS -DTHRUST_DEBUG $(INCLUDE) $(CUDA_LIBS) $(SRC_DIR)/main.cu -o main

timing:
	$(CUDA) $(CUDAFLAGS) -D=TIMING $(INCLUDE) $(CUDA_LIBS) $(SRC_DIR)/main.cu -o main

clean:
	$(RM) main
