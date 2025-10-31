#pragma once

#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_math.h>


#define PE_CUDA_BLOCKS 256

PHYS_NAMESPACE_BEGIN

/**
 * @brief Initializes CUDA.
 * 
 * @param argc The number of command-line arguments.
 * @param argv The command-line arguments.
 */
inline void cudaInit(int argc, char **argv){
    int devID;

    // use command-line specified CUDA device, otherwise use device with highest Gflops/s
    devID = findCudaDevice(argc, (const char **)argv);

    // Check if a CUDA device is found
    if (devID < 0){
        printf("No CUDA Capable devices found, exiting...\n");
        exit(EXIT_SUCCESS);
    }
}

//// follow helper_cuda.h

/**
 * @brief Checks for CUDA errors and exits if an error is found.
 * 
 * @param err The CUDA error code.
 * @param file The file name where the error occurred.
 * @param line The line number where the error occurred.
 */
#define checkCudaError(err) __checkCudaError(err, __FILE__, __LINE__)
inline void __checkCudaError(cudaError_t err, const char *file, const int line){
    if (cudaSuccess != err) {
        fprintf(stderr,
                "%s(%i) : CudaError()"
                " %s : (%d) %s.\n",
                file, line, static_cast<int>(err),
                cudaGetErrorString(err));
        exit(-1);
    }
}

/**
 * @brief Check for CUDA errors and exit the program if an error occurs.
 * 
 * @param errMsg - The error message to display.
 * @param file - The name of the file where the error occurred.
 * @param line - The line number where the error occurred.
 */
#define checkCuda(msg) __checkCuda(msg, __FILE__, __LINE__)
inline void __checkCuda(const char *errMsg, const char *file, const int line){
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr,
                "%s(%i) : getLastCudaError() CUDA error :"
                " %s : (%d) %s.\n",
                file, line, errMsg, static_cast<int>(err),
                cudaGetErrorString(err));
        exit(-1);
    }
}

/**
 * @brief Compute the number of CUDA threads and blocks.
 * 
 * @param n The total number of elements.
 * @param bs The desired block size.
 * @param nb The number of blocks (output).
 * @param nt The number of threads per block (output).
 */
inline void computeCudaThread(unsigned int n, unsigned int bs, unsigned int &nb, unsigned int &nt){
    // Determine the number of threads per block
    nt = min(bs, n);//// #thread

    // Determine the number of blocks
    nb = (n % nt != 0) ? (n / nt + 1) : (n / nt);//// #block
}

#ifdef __INTELLISENSE__
#define CUDA_KERNEL(...)
#else
#define CUDA_KERNEL(...) <<< __VA_ARGS__ >>>
#endif

PHYS_NAMESPACE_END

