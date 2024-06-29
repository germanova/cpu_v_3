#include "params_CPU.hpp"


Matrix init(int mat_size, std::vector<int>& initSpins)
{ // to initialize the matrix is used loop unrolling. To deal with remainders
    // is ensured that mat_size is a multiple of 8, the number of operations per iteration
    if (mat_size % 8 != 0)
    {
        mat_size = std::floor(mat_size / 8.0) * 8.0;
        std::cout << "truncated mat_size: " << mat_size;
        std::cout << "\n";
    }
    // create Matrix
    Matrix mat(mat_size, mat_size);

    // assign random values (+1 or -1) to the matrix
    int k = 0;
    for (int i = 0; i < mat_size; i++)
    {
        for (int j = 0; j < mat_size; j += 8)
        {
            // loop unrolling
            mat.at(i, j) = initSpins[k];
            mat.at(i, j + 1) = initSpins[k + 1];
            mat.at(i, j + 2) = initSpins[k + 2];
            mat.at(i, j + 3) = initSpins[k + 3];
            mat.at(i, j + 4) = initSpins[k + 4];
            mat.at(i, j + 5) = initSpins[k + 5];
            mat.at(i, j + 6) = initSpins[k + 6];
            mat.at(i, j + 7) = initSpins[k + 7];
        }
        k += 8;
    }

    return mat;
}

// a function to get the energy difference using nearest neighbours sum
int diff(Matrix& mat, int i, int j)
{
    int current_position = mat.at(i, j);
    int nearest_neighbors = 0;

    // here its assumed that when the spin is on the borders it will have only a subset
    // of neighbours according to its position. A spin on the upper right corner (0,0)
    // will only have two neighbors 0,1 ; 1,0
    if (i == 0)
    {
        nearest_neighbors += mat.at(i + 1, j);
    }
    else if (i == mat_size - 1)
    {
        nearest_neighbors += mat.at(i - 1, j);
    }
    else
    {
        nearest_neighbors += mat.at(i + 1, j);
        nearest_neighbors += mat.at(i - 1, j);
    }

    if (j == 0)
    {
        nearest_neighbors += mat.at(i, j + 1);
    }
    else if (j == mat_size - 1)
    {
        nearest_neighbors += mat.at(i, j - 1);
    }
    else
    {
        nearest_neighbors += mat.at(i, j + 1);
        nearest_neighbors += mat.at(i, j - 1);
    }

    int nn_operation = current_position * nearest_neighbors;

    return nn_operation;
}
// a function to print the final state of the matrix
void print_state(Matrix mat, int mat_size)
{
    // print matrix
    for (int i = 0; i < mat_size; ++i)
    {
        for (int j = 0; j < mat_size; ++j)
        {
            if (mat.at(i, j) == 1)
            {
                std::cout << "+ ";
            }
            else
            {
                std::cout << "- ";
            }
        }
        std::cout << "\n";
    }
    std::cout << "\n";

}

void mean_energy(Matrix& mat, int mat_size)
{
    double sum = 0.0;
    for (int i = 0; i < mat_size; i++) {
        for (int j = 0; j < mat_size; j++)
        {
            sum += mat.at(i, j);
        }
    }

    int total_elements = mat_size * mat_size;
    double mean = sum / (total_elements);
    std::cout << "Mean of " << total_elements << " samples: " << mean << std::endl;
}

ThreadRange thread_range(int sample_size, int num_threads, int thread_id)
{
    ThreadRange result;
    int chunk_size = sample_size / num_threads;
    int start = thread_id * chunk_size;
    int end = (thread_id == num_threads - 1) ? sample_size : (start + chunk_size);

    result.start = start;
    result.end = end;
    return result;
}


Matrix submatrix(Matrix mat, int start_row, int end_row, int mat_size)
{

    int rows = end_row - start_row + 1;
    Matrix sub_mat(rows, mat_size);

    for (int i = start_row; i <= end_row; i++) {
        for (int j = 0; j <= mat_size - 1; j++) {
            sub_mat.at(i - start_row, j) = mat.at(i, j);
        }
    }

    return sub_mat;
}

void write(Matrix& mat, Matrix sub_mat, int start_row, int end_row, int mat_size)
{
    for (int i = start_row; i <= end_row; i++) {
        for (int j = 0; j <= mat_size - 1; j++) {
            mat.at(i, j) = sub_mat.at(i - start_row, j);
        }
    }
}



