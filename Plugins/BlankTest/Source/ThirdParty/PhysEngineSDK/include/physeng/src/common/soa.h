#pragma once
#include <functional>
#include "common/array.h"


PHYS_NAMESPACE_BEGIN

//// after

/// @brief Declare a GPU kernel template.
/// @param func_name The name of the function.
/// @param ... The arguments of the function.
#define DECLARE_GPU_KERNEL_TEMP(func_name,...)\
    template __device__ void _k_##func_name<MemType::GPU>(int i, __VA_ARGS__);

/// @brief Fill the call to the GPU device code.
/// @param func_name The name of the function.
/// @param ... The arguments of the function.
#define FILL_CALL_GPU_DEVICE_CODE(func_name,...)\
        {\
        unsigned int nb=0, nt=0;\
        computeCudaThread(size, PE_CUDA_BLOCKS, nb, nt);\
        _g_##func_name<<<nb,nt>>>(size, __VA_ARGS__);\
        getLastCudaError("arrayFill kernel failed");\
        }
    
    
/// @brief Declare a kernel template for both CPU and GPU.
/// @param func_name The name of the function.
/// @param ... The arguments of the function.
#define DECLARE_KERNEL_TEMP(func_name,...)\
    template __host__ __device__ void _k_##func_name<MemType::CPU>(int i, __VA_ARGS__);\
    template __host__ __device__ void _k_##func_name<MemType::GPU>(int i, __VA_ARGS__);

/// @brief Fill the call to the device code based on the memory type.
/// @param func_name The name of the function.
/// @param ... The arguments of the function.
#define FILL_CALL_DEVICE_CODE(func_name,...)\
    if constexpr(MT==MemType::GPU){\
        unsigned int nb=0, nt=0;\
        computeCudaThread(size, PE_CUDA_BLOCKS, nb, nt);\
        _g_##func_name<<<nb,nt>>>(size, __VA_ARGS__);\
        getLastCudaError("arrayFill kernel failed");\
    } else {\
        __pragma(omp parallel for) \
        for(unsigned int i=0;i<size;i++) _k_##func_name<MemType::CPU>(i, __VA_ARGS__);}

// fix bug here
/// @brief Check if the index is valid.
/// @param size The size of the array.
#define IF_IDX_VALID(size)\
    int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < size)

/// @brief Declare a CPU kernel template.
/// @param func_name The name of the function.
/// @param ... The arguments of the function.
#define DECLARE_CALL_CPU_KERNEL(func_name,...)\
    template void c_##func_name<MemType::CPU>(int size, __VA_ARGS__);

/// @brief Declare a GPU kernel template.
/// @param func_name The name of the function.
/// @param ... The arguments of the function.
#define DECLARE_CALL_GPU_KERNEL(func_name,...)\
    template void c_##func_name<MemType::GPU>(int size, __VA_ARGS__);


//// some common kernel function
/**
 * @brief Fill an array with a target value.
 * @param size The size of the array.
 * @param tar The target value to fill the array with.
 * @param vx The array to be filled.
 */
template<MemType MT>
void callFillArray(int size, vec3r tar, VecArray<vec3r,MT>& vx);

/**
 * @brief Add two arrays element-wise.
 * @param size The size of the arrays.
 * @param a The first array.
 * @param b The second array.
 */
template<MemType MT>
void callAddArray(int size, VecArray<vec3r,MT>& a, VecArray<vec3r,MT>& b);

/**
 * @brief Add a target value to an array element-wise.
 * @param size The size of the array.
 * @param tar The target value to add.
 * @param vx The array to add the target value to.
 */
template<MemType MT>
void callAddArray(int size, vec3r tar, VecArray<vec3r,MT>& vx);


PHYS_NAMESPACE_END