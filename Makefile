BIN := 1dtest

CC=gcc
CXX=g++
RM=rm -f

CUDA_INSTALL_PATH=/usr/local/cuda-7.5
CUDA=$(CUDA_INSTALL_PATH)/bin/nvcc
CUDA_LIBS=-L"$(CUDA_INSTALL_PATH)/lib64"

CXXFLAGS=-g -O3 -Wall
CUDAFLAGS=-O3 -arch=compute_35
#CUDAFLAGS=-g -G -O3 -arch=compute_35
#CUDAFLAGS=-g -G -O3 -arch=compute_35 --ptxas-options=-v

INCLUDE= -I"$(CUDA_INSTALL_PATH)/include"
SRC_DIR=src

all: 
	$(CUDA) $(CUDAFLAGS) $(INCLUDE) $(CUDA_LIBS) $(SRC_DIR)/main.cu -o main

clean:
	$(RM) main
