#pragma once

/**
 * @brief This file provides common macros
 */

/**
 * namespace
 */
#define PHYS_NAMESPACE_BEGIN namespace physeng {
#define PHYS_NAMESPACE_END }
#define USE_PHYS_NAMESPACE using namespace physeng;

/**
 * @brief Macro for specifying 16-byte alignment
 * @param a The variable or type to be aligned
 */
#define ATTRIBUTE_ALIGNED16(a) a alignas(16)
/**
 * @brief Macro for specifying 64-byte alignment
 * @param a The variable or type to be aligned
 */
#define ATTRIBUTE_ALIGNED64(a) a alignas(64)
/**
 * @brief Macro for specifying 128-byte alignment
 * @param a The variable or type to be aligned
 */
#define ATTRIBUTE_ALIGNED128(a) a alignas(128)

/**
 * filename macro
 */
#ifdef __APPLE__
#define __FILENAME__ (strrchr(__FILE__, '/') + 1)
#else
#define __FILENAME__ (strrchr(__FILE__, '\\') + 1)
#endif

/**
 * inline
 */
#ifdef _WIN32
    #define FORCE_INLINE __forceinline
#else
    #define FORCE_INLINE __attribute__ ((always_inline)) inline
#endif

/**
 * @brief Macro for defining a member variable with corresponding setter and getter functions.
 * 
 * This macro defines a member variable with the specified type and name, and also generates
 * corresponding setter and getter functions for accessing and modifying the value of the member.
 * 
 * @param T The type of the member variable.
 * @param name The name of the member variable.
 * @param Name The capitalized name of the member variable.
 */
#define DEFINE_MEMBER_SET_GET(T, name, Name)\
    protected:\
    T name;\
    public:\
    void set##Name(const T& t){ name = t; }\
    const T& get##Name() const { return name; }

#define DEFINE_MEMBER_GET(T, name, Name)\
    protected:\
    T name;\
    public:\
    const T& get##Name() const { return name; }

/**
 * @brief Macro for defining a member variable with corresponding setter and getter functions for a pointer type.
 * 
 * This macro defines a member variable with the specified pointer type and name, and also generates
 * corresponding setter and getter functions for accessing and modifying the value of the member.
 * 
 * @param T The pointer type of the member variable.
 * @param name The name of the member variable.
 * @param Name The capitalized name of the member variable.
 */
#define DEFINE_MEMBER_PTR_SET_GET(T, name, Name)\
    protected:\
    T* name;\
    public:\
    void set##Name(T* t){ name = t; }\
    T* get##Name() const { return name; }

#define DEFINE_MEMBER_PTR_GET(T, name, Name)\
    protected:\
    T* name;\
    public:\
    T* get##Name() const { return name; }

/**
 * @brief Following macros define a common pattern for defining array member, setter, and getter functions.
 *
 * @param T     The type of the array member
 * @param cnt   The number of elements in the array
 * @param name  The name of the array member
 * @param Name  The capitalized name of the array member
 */
#define DEFINE_ARRAY_MEMBER_SET_GET(T, cnt, name, Name)\
    protected:\
    T name[cnt];\
    public:\
    void set##Name(int idx, const T& t){ name[idx] = t; }\
    const T& get##Name(int idx) const { return name[idx]; }

//// soa in array.h
//// use only in particle array
// #define DEFINE_DYNAMIC_ARRAY_MEMBER_SET_GET(T, name, Name)\
//     protected:\
//     ObjectArray<T> name;\
//     public:\
//     void set##Name(int idx, const T& t){ name[idx] = t; }\
//     const T& get##Name(int idx) const { return name[idx]; }

/**
 * assert
 */
#include <assert.h>
#define ASSERT assert

/**
 * @brief Custom assertion macro
 * 
 * This macro is used to assert a condition and display a custom error message
 * with additional information such as the time, file name, and line number.
 * 
 * @param cond The condition to be asserted
 * @param str  The custom error message to be displayed
 */
#define CUSTOM_ASSERT(cond, str)\
    if(cond){\
        std::cout << "\033[35m[" << __TIME__ << "][" << __FILENAME__ << ":" << __LINE__ << "][Assert]\033[0m " << str << std::endl;\
        exit(-1);\
    }

/**
 * @brief Custom assertion macro for unimplemented code
 * 
 * This macro is used to assert that a particular code path has not been implemented yet.
 * It is typically used as a placeholder in the code to remind the developer to implement
 * the missing functionality.
 * 
 * @note This macro should be used sparingly and removed once the code has been implemented.
 */
#define UNIMPL_ASSERT()\
    CUSTOM_ASSERT(true, "unimplemented")

#if defined(DEBUG) || defined (_DEBUG)
    #define DEBUG_ASSERT assert
#else
    #define DEBUG_ASSERT void  // suppress compiler unused-value warning
#endif

//// cuda related macro
#ifdef PE_USE_CUDA
//// TOFIXMACRO
// #if true
    #include <cuda_runtime.h>
    #define PE_CUDA_FUNC __host__ __device__
    #define PE_KERNEL_FUNC __host__ __device__
    enum MemType{
        CPU=0, //// cpu function
        GPU=1  //// gpu function
    };
#else
    #define PE_CUDA_FUNC
    #define PE_KERNEL_FUNC
    enum MemType{
        CPU=0, //// cpu function
        GPU=1  //// gpu function
    };
#endif

//// refer to [](https://stackoverflow.com/questions/56717411/control-conditional-openmp-in-define-macro)
#ifdef PE_USE_OMP
    //// TODO: modified into our method
    #include <omp.h>     // This line won't add the library if you don't compile with -fopenmp option.
    #ifdef _MSC_VER
        // For Microsoft compiler
        #define OMP_FOR(n) __pragma(omp parallel for if(n>10)) 
        #define iterate_cell_idx(i,grid) \
            __pragma(omp parallel for if(grid.Number_Of_Cells()>=10)) \
            for(int i=0;i<grid.Number_Of_Cells();i++)
        #define iterate_node_idx(i,grid) \
            __pragma(omp parallel for if(grid.Number_Of_Nodes()>=10)) \
            for(int i=0;i<grid.Number_Of_Nodes();i++)
        #define iterate_face_idx(axis,i,grid) \
            for(int axis=0;axis<d;axis++) \
            __pragma(omp parallel for if(grid.Number_Of_Faces(axis)>=10)) \
            for(int i=0;i<grid.Number_Of_Faces(axis);i++)
    #else  // assuming "__GNUC__" is defined
        // For GCC compiler
        #define OMP_FOR(n) _Pragma("omp parallel for if(n>10)")
        #define iterate_cell_idx(i,grid) \
            _Pragma("omp parallel for") \
            for(int i=0;i<grid.Number_Of_Cells();i++)
        #define iterate_node_idx(i,grid) \
            _Pragma("omp parallel for") \
            for(int i=0;i<grid.Number_Of_Nodes();i++)
        #define iterate_face_idx(axis,i,grid) \
            for(int axis=0;axis<d;axis++) \
            _Pragma("omp parallel for") \
            for(int i=0;i<grid.Number_Of_Faces(axis);i++)
    #endif
#else
    #define omp_get_thread_num() 0
    #define OMP_FOR(n)
    #define iterate_cell_idx(i,grid) for(int i=0;i<grid.Number_Of_Cells();i++)
    #define iterate_node_idx(i,grid) for(int i=0;i<grid.Number_Of_Nodes();i++)
    #define iterate_face_idx(axis,i,grid) for(int axis=0;axis<d;axis++)for(int i=0;i<grid.Number_Of_Faces(axis);i++)
#endif