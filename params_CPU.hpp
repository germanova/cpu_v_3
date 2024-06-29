#ifndef PARAMS
#define PARAMS

#include <vector>
#include <random>
#include <iostream>
#include <cmath>
#include <map>
#include <algorithm>
#include <iostream>
#include <mkl_vsl.h>
#include "mkl.h"
//#include "mkl_omp_offload.h"
#include <omp.h>
#include <chrono>

// number of threads
//const int num_threads = 4;
const int num_threads = omp_get_max_threads();
// number of monte carlo iterations
const int N_iter = 10000000;
// number of monte carlo iterations per thread
const int iterations_per_thread = N_iter / num_threads;
// 1/temperature for strength reduction
const float T_inv = 1.0f / 1.0f;
// lattice size
const int mat_size = 32;
// rows per sub matrix
const int ROWS = mat_size / num_threads;
// seeds
const int init_seed = 10;
const int row_seed = 100;
const int col_seed = 200;
const int threshold_seed = 17;


// a struct for defining the Matrix taken from lecture 0 just slighly modified so that it only recieve one parameter
struct Matrix
{
    int ROWS;
    int COLS;
    std::vector<int> data;

    // move matrix init
    Matrix(int rows, int cols) : ROWS(rows), COLS(cols), data(rows* cols, 0.) {}

    int& at(int i, int j)
    {
        return data[i * COLS + j];
    }
};

struct ThreadRange {
    int start;
    int end;
};


Matrix init(int mat_size, std::vector<int>& initSpins);
int diff(Matrix& mat, int i, int j);
void print_state(Matrix mat, int mat_size);
void mean_energy(Matrix& mat, int mat_size);
Matrix submatrix(Matrix mat, int start_row, int end_row, int mat_size);
void write(Matrix& mat, Matrix sub_mat, int start_row, int end_row, int mat_size);
ThreadRange thread_range(int sample_size, int num_threads, int thread_id);
// end the if stated at the start
#endif