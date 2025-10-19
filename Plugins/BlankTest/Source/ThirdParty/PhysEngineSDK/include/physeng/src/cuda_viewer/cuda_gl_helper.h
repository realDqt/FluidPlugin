#pragma once

// OpenGL Graphics includes
/**
@brief This is a helper_gl function.

This function does something useful.

@param arg1 The first parameter.
@param arg2 The second parameter.
@return The result of the function.
*/
#include <helper_gl.h>
#if defined (WIN32)
#include <GL/wglew.h>
#endif
#if defined(__APPLE__) || defined(__MACOSX)
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
  #include <GLUT/glut.h>
  #ifndef glutCloseFunc
  #define glutCloseFunc glutWMCloseFunc
  #endif
#else
#include <GL/freeglut.h>
#endif

// CUDA runtime
#include <cuda_runtime.h>

// CUDA utilities and system includes
#include <helper_functions.h>
#include <helper_cuda.h>    // includes cuda.h and cuda_runtime_api.h
/**
@brief This is a helper_math function.

This function performs mathematical operations.
*/
#include <helper_math.h>

/**
@brief This is a CUDA runtime function.

This function performs CUDA operations.

*/
#include <cuda_runtime.h>

/**
@brief This is a CUDA GL interop function.

This function performs CUDA GL interop operations.
*/
#include <cuda_gl_interop.h>


// This function unregisters a GL buffer object
void registerGLBufferObject(uint vbo, struct cudaGraphicsResource **cuda_vbo_resource);
// This function maps a GL buffer object for CUDA access
void unregisterGLBufferObject(struct cudaGraphicsResource *cuda_vbo_resource);
/**
 * Maps a GL buffer object for CUDA access.
 *
 * @param cuda_vbo_resource - pointer to the CUDA graphics resource
 * @return a pointer to the mapped memory
 */
/**
@brief Maps a GL buffer object for CUDA access.

This function maps a GL buffer object for CUDA access.

@param cuda_vbo_resource Pointer to the CUDA graphics resource.
@return A pointer to the mapped memory.
*/
void *mapGLBufferObject(struct cudaGraphicsResource **cuda_vbo_resource);

/**
@brief Unmaps a GL buffer object.

This function unmaps a GL buffer object.

@param cuda_vbo_resource Pointer to the CUDA graphics resource.
*/
void unmapGLBufferObject(struct cudaGraphicsResource *cuda_vbo_resource);
