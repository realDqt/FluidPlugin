#pragma once

#ifdef _WIN32
#include <stdlib.h>
#else
#include <cstdlib>
#endif

#include "math/math.h"
#include "common/logger.h"
#include "common/cuda/cuda_util.h"
#include "common/cuda/cuda_math.h"

#ifdef PE_USE_CUDA
#include "helper_cuda.h"
#endif

PHYS_NAMESPACE_BEGIN

/**
 * @brief Allocates memory with alignment using the C11 standard function.
 *
 * @param size The size of the memory to allocate.
 * @param align The alignment of the memory.
 * @return A pointer to the allocated memory.
 */
inline void* alignedAllocC11Std(size_t size, size_t align) {
    // Calculate the mask based on the alignment
    size_t mask = align - 1;

    // Check if the size has a remainder when divided by the mask
    const size_t remainder = size & mask;

    // If there is a remainder, adjust the size to the next aligned value
    if(remainder != 0)
        size = size & !mask + align;
#ifdef _WIN32
    // Allocate aligned memory using _aligned_malloc on Windows
    void* ptr = _aligned_malloc(size, align);
#else
    // TODO FIX ME segment fault using aligned alloc, fallback to normal one for now
    // Allocate memory using malloc on other platforms
    void* ptr = malloc(size);//, align);
#endif
    // LOG_OSTREAM_WARN << "malloc " << size << " bytes at " << ptr << std::endl;
    return ptr;
}

/**
 * @brief Frees a memory block allocated with aligned allocation.
 * 
 * @param ptr The pointer to the memory block.
 * @param align The alignment of the allocation.
 */
inline void alignedFreeC11Std(void* ptr, size_t align) {
    // LOG_OSTREAM_WARN << "free at " << ptr << std::endl;
#ifdef _WIN32
	_aligned_free(ptr);
#else
    free(ptr);
#endif    
}

/**
 * @brief Allocate an array of memory with a specific type and size.
 * 
 * @tparam T The type of the elements in the array.
 * @tparam MT The memory type for the allocation.
 * @param ptr A pointer to store the allocated memory.
 * @param size The size of the array to allocate.
 */
template<typename T, MemType MT> 
void allocArray(T** ptr, unsigned int size){
    // Allocate memory based on the memory type
    if constexpr(MT==MemType::CPU)
        *ptr = (T*)malloc(size * sizeof(T));
    #ifdef PE_USE_CUDA
    if constexpr(MT==MemType::GPU)
        physeng::checkCudaError(cudaMalloc(ptr, size*sizeof(T)));
    #else
    if constexpr(MT==MemType::GPU)
        LOG_OSTREAM_ERROR<<"No CUDA Module"<<std::endl;
    #endif
};

/**
 * @brief Frees an array of memory allocated with a specific type and memory type.
 * 
 * @tparam T The type of the elements in the array.
 * @tparam MT The memory type of the allocation.
 * @param ptr A pointer to the memory block.
 */
template<typename T, MemType MT> 
void freeArray(T** ptr){
    if constexpr(MT==MemType::CPU){
        free(*ptr); ptr=nullptr; 
        return;
    }
    #ifdef PE_USE_CUDA
    if constexpr(MT==MemType::GPU){
        physeng::checkCudaError(cudaFree(*ptr)); ptr=nullptr; 
        return;
    }
    #else
    if constexpr(MT==MemType::GPU){
        LOG_OSTREAM_ERROR<<"No CUDA Module"<<std::endl;
        return;
    }
    #endif
};


//// fill
//// fill helper
// #ifdef PE_USE_CUDA
template<typename T>
void fillArrayCuda(T** ptr, const T& t, unsigned int size);

/**
 * @brief Fills an array with a given value.
 * 
 * @tparam T The type of the array elements.
 * @tparam MT The memory type (CPU or GPU).
 * @param ptr Pointer to the array.
 * @param t The value to fill the array with.
 * @param size The size of the array.
 */
template<typename T, MemType MT>
void fillArray(T** ptr, const T& t, int size){
    // Fill the array with the given value for CPU memory type
    if constexpr(MT==MemType::CPU){
        for (unsigned int i = 0; i < size; i++) (*ptr)[i] = t;
        return;
    }
    #ifdef PE_USE_CUDA
    // Fill the array with the given value for GPU memory type
    if constexpr(MT==MemType::GPU){
        fillArrayCuda(ptr,t,size);
        return;
    }
    #else
    // Log an error message for GPU memory type when CUDA module is not available
    if constexpr(MT==MemType::GPU){
        LOG_OSTREAM_ERROR<<"No CUDA Module"<<std::endl;
        return;
    }
    #endif
}

/**
 * @brief Copy an array from one memory type to another.
 *
 * @tparam T The type of the elements in the array.
 * @tparam MT1 The memory type of the source array.
 * @tparam MT2 The memory type of the destination array.
 * @param ptr1 Pointer to the source array.
 * @param ptr2 Pointer to the destination array.
 * @param size The size of the array.
 */
template<typename T, MemType MT1, MemType MT2>
void copyArray(T** ptr1, T** ptr2, unsigned int size){
    // Copy array from host to host
    if constexpr(MT1==MemType::CPU && MT2==MemType::CPU){
        // LOG_OSTREAM_DEBUG<<"copyArray h->h"<<std::endl;
        memcpy(*(ptr1),*(ptr2), size*sizeof(T)); 
        return;
    }
    #ifdef PE_USE_CUDA
    // Copy array from host to device
    if constexpr(MT1==MemType::GPU && MT2==MemType::CPU){
        // LOG_OSTREAM_DEBUG<<"copyArray h->d"<<std::endl;
        physeng::checkCudaError(cudaMemcpy(*(ptr1),*(ptr2), size*sizeof(T), cudaMemcpyHostToDevice)); 
        return;
    }
    // Copy array from device to host
    if constexpr(MT1==MemType::CPU && MT2==MemType::GPU){
        // LOG_OSTREAM_DEBUG<<"copyArray d->h"<<std::endl;
        physeng::checkCudaError(cudaMemcpy(*(ptr1),*(ptr2), size*sizeof(T), cudaMemcpyDeviceToHost)); 
        return;
    }
    // Copy array from device to device
    if constexpr(MT1==MemType::GPU && MT2==MemType::GPU){
        // LOG_OSTREAM_DEBUG<<"copyArray d->d"<<std::endl;
        physeng::checkCudaError(cudaMemcpy(*(ptr1),*(ptr2), size*sizeof(T), cudaMemcpyDeviceToDevice));
        return;
    }
    #else
    // No CUDA module, cannot copy arrays
    { LOG_OSTREAM_ERROR<<"No CUDA Module"<<std::endl; return; }
    #endif
    
};

/**
 * @brief Copies elements from one array to another.
 *
 * @tparam T The type of the elements in the array.
 * @tparam MT1 The memory type of the source array.
 * @tparam MT2 The memory type of the destination array.
 * @param ptr1 Pointer to the source array.
 * @param ptr2 Pointer to the destination array.
 * @param start The starting index of the elements to copy.
 * @param size The number of elements to copy.
 */
template<typename T, MemType MT1, MemType MT2>
void copyArray(T** ptr1, T** ptr2, unsigned int start, unsigned int size){
    // Copy elements from host to host
    if constexpr(MT1==MemType::CPU && MT2==MemType::CPU){
        LOG_OSTREAM_DEBUG<<"copyArray h->h"<<std::endl;
        memcpy(*(ptr1)+start,*(ptr2)+start, size*sizeof(T)); 
        return;
    }
    #ifdef PE_USE_CUDA
    // Copy elements from host to device
    if constexpr(MT1==MemType::GPU && MT2==MemType::CPU){
        // LOG_OSTREAM_DEBUG<<"copyArray h->d"<<std::endl;
        physeng::checkCudaError(cudaMemcpy(*(ptr1)+start,*(ptr2)+start, size*sizeof(T), cudaMemcpyHostToDevice)); 
        return;
    }
    // Copy elements from device to host
    if constexpr(MT1==MemType::CPU && MT2==MemType::GPU){
        // LOG_OSTREAM_DEBUG<<"copyArray d->h"<<std::endl;
        physeng::checkCudaError(cudaMemcpy(*(ptr1)+start,*(ptr2)+start, size*sizeof(T), cudaMemcpyDeviceToHost)); 
        return;
    }
    // Copy elements from device to device
    if constexpr(MT1==MemType::GPU && MT2==MemType::GPU){
        // LOG_OSTREAM_DEBUG<<"copyArray d->d"<<std::endl;
        physeng::checkCudaError(cudaMemcpy(*(ptr1)+start,*(ptr2)+start, size*sizeof(T), cudaMemcpyDeviceToDevice)); 
        return;
    }
    #else
    // No CUDA module available
    { LOG_OSTREAM_ERROR<<"No CUDA Module"<<std::endl; return; }
    #endif
    
};

/**
 * @brief Copy an array from one memory type to another.
 *
 * @tparam T The type of the elements in the array.
 * @tparam MT1 The memory type of the source array.
 * @tparam MT2 The memory type of the destination array.
 * @param ptr1 Pointer to the source array.
 * @param ptr2 Pointer to the destination array.
 * @param start1 Starting index in the source array.
 * @param start2 Starting index in the destination array.
 * @param size The number of elements to copy.
 */
template<typename T, MemType MT1, MemType MT2>
void copyArray(T** ptr1, T** ptr2, unsigned int start1, unsigned int start2, unsigned int size) {
    // Copy elements from host to host
    if constexpr (MT1 == MemType::CPU && MT2 == MemType::CPU) {
        LOG_OSTREAM_DEBUG << "copyArray h->h" << std::endl;
        memcpy(*(ptr1)+start1, *(ptr2)+start2, size * sizeof(T));
        return;
    }
#ifdef PE_USE_CUDA
    // Copy elements from host to device
    if constexpr (MT1 == MemType::GPU && MT2 == MemType::CPU) {
        // LOG_OSTREAM_DEBUG<<"copyArray h->d"<<std::endl;
        physeng::checkCudaError(cudaMemcpy(*(ptr1)+start1, *(ptr2)+start2, size * sizeof(T), cudaMemcpyHostToDevice));
        return;
    }
    // Copy elements from device to host
    if constexpr (MT1 == MemType::CPU && MT2 == MemType::GPU) {
        // LOG_OSTREAM_DEBUG<<"copyArray d->h"<<std::endl;
        physeng::checkCudaError(cudaMemcpy(*(ptr1)+start1, *(ptr2)+start2, size * sizeof(T), cudaMemcpyDeviceToHost));
        return;
    }
    // Copy elements from device to device
    if constexpr (MT1 == MemType::GPU && MT2 == MemType::GPU) {
        // LOG_OSTREAM_DEBUG<<"copyArray d->d"<<std::endl;
        physeng::checkCudaError(cudaMemcpy(*(ptr1)+start1, *(ptr2)+start2, size * sizeof(T), cudaMemcpyDeviceToDevice));
        return;
    }
#else
    // No CUDA module, cannot copy arrays
    { LOG_OSTREAM_ERROR << "No CUDA Module" << std::endl; return; }
#endif

};


PHYS_NAMESPACE_END