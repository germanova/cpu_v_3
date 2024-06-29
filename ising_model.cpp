#include "params_CPU.hpp"


// Parallel Ising Model with MKL streams for random number generation and OpenMP
int main()
{
    auto start = std::chrono::high_resolution_clock::now();
    // create a lookup table with the 9 possible values of the change in energy
    float LookUpTable[9] = {
        std::exp(8.0f * T_inv), // -8
        std::exp(6.0f * T_inv), // -6
        std::exp(4.0f * T_inv), // -4
        std::exp(2.0f * T_inv), // -2
        1.0f,                   // 0
        std::exp(-2.0f * T_inv),
        std::exp(-4.0f * T_inv),
        std::exp(-6.0f * T_inv),
        std::exp(-8.0f * T_inv) };

    // Initialize MKL
    mkl_set_num_threads_local(num_threads); // Set the number of threads

    int matrix_init_spins = mat_size * mat_size;
    std::vector<int> initSpins(matrix_init_spins);

    // Initialize MKL RNG for integers, source of random numbers
    VSLStreamStatePtr initStream;
    vslNewStream(&initStream, VSL_BRNG_MT19937, init_seed);

    int* rowSpins = static_cast<int*>(mkl_malloc(N_iter * sizeof(int), 64)); // 64-byte alignment
    //std::vector<int> randomSpins(random_spins);
    // Initialize MKL RNG for integers
    VSLStreamStatePtr rowStream;
    vslNewStream(&rowStream, VSL_BRNG_MT19937, row_seed);

    int* colSpins = static_cast<int*>(mkl_malloc(N_iter * sizeof(int), 64)); // 64-byte alignment
    //std::vector<int> randomSpins(random_spins);
    // Initialize MKL RNG for integers
    VSLStreamStatePtr colStream;
    vslNewStream(&colStream, VSL_BRNG_MT19937, col_seed);

    // Vector for random numbers allocation
    float* randomThreshold = static_cast<float*>(mkl_malloc(N_iter * sizeof(float), 64));
    //std::vector<float> randomThreshold(N_iter);
    // Initialize MKL RNG for floats
    VSLStreamStatePtr thresholdStream;
    vslNewStream(&thresholdStream, VSL_BRNG_MT19937, threshold_seed);

    std::cout << "num threads: " << num_threads << std::endl;

    // Parallelize random number generation using OpenMP
#pragma omp parallel num_threads(num_threads)
    {
        // Get the thread ID
        int thread_id = omp_get_thread_num();

        // Calculate the range of random numbers to generate for this thread
        ThreadRange range = thread_range(matrix_init_spins, num_threads, thread_id);

        //std::cout << "Thread:  " << thread_id << " of: " << omp_get_num_threads() << std::endl;
        // Generate random numbers for the specified range
        viRngUniform(VSL_RNG_METHOD_UNIFORM_STD, initStream, range.end - range.start, initSpins.data() + range.start, 0, 2);

        std::replace_if(initSpins.begin() + range.start, initSpins.begin() + range.end,
            [](int& i) { return i == 0; }, -1);

        // Calculate the range of random spins to generate for this thread
        range = thread_range(N_iter, num_threads, thread_id);

        // Generate random integers
        viRngUniform(VSL_RNG_METHOD_UNIFORM_STD, rowStream, range.end - range.start, rowSpins + range.start, 0, mat_size);
        viRngUniform(VSL_RNG_METHOD_UNIFORM_STD, colStream, range.end - range.start, colSpins + range.start, 0, mat_size);

        // Generate random floats
        vsRngUniform(VSL_RNG_METHOD_UNIFORM_STD, thresholdStream, range.end - range.start, randomThreshold + range.start, 0.0, 1.0);
    }


    vslDeleteStream(&initStream);
    vslDeleteStream(&thresholdStream);
    vslDeleteStream(&rowStream);
    vslDeleteStream(&colStream);

    // initialize the matriz with random values (-1 or 1)
    Matrix mat = init(mat_size, initSpins);
    if (mat_size <= 64)
    {
        //  a function to print the final state of the matrix
        print_state(mat, mat_size);
    }

    mean_energy(mat, mat_size);


#pragma omp parallel for schedule(dynamic) num_threads(num_threads)
    for (int k = 0; k < N_iter; ++k)
    {
        // select a random spin
        int i = rowSpins[k];
        int j = colSpins[k];

        // energy difference
        // common subexpression
        int dE = diff(mat, i, j);

        // spin criteria following literature
        if (dE <= 0 || randomThreshold[k] < LookUpTable[dE + 4])
        {
            mat.at(i, j) *= -1;
        }
    }

    mkl_free(rowSpins); // Free the memory allocated with mkl_malloc
    mkl_free(colSpins);
    mkl_free(randomThreshold);

    if (mat_size <= 64)
    {
        //  a function to print the final state of the matrix
        print_state(mat, mat_size);
        // print the mean of energy in the lattice
        mean_energy(mat, mat_size);
    }
    else
    {
        mean_energy(mat, mat_size);
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "Time taken by function: " << duration.count() / 1000000 << " seconds" << std::endl;

}

