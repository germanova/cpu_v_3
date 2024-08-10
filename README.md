# Parallel Implementation of the Ising Model in C++ using OpenMP and Intel MKL Library

Parallel implementation of the Ising Model using C++ with the support of the Intel MKL library for efficient random number generation and OpenMP for parallelization. The Ising Model is a mathematical model of ferromagnetism in statistical mechanics, and this implementation aims to leverage modern computing resources to perform large-scale simulations efficiently.


- **Random Number Generation**: Utilizes the Intel MKL library for random number generation.
- **Parallel Processing**: Employs OpenMP to parallelize random number generation and lattice updates.

## Prerequisites

- **C++ Compiler**: A compliant C++ compiler (e.g., g++, clang++).
- **Intel MKL Library**: Intel Math Kernel Library for random number generation.
- **OpenMP**: OpenMP support for parallel processing.