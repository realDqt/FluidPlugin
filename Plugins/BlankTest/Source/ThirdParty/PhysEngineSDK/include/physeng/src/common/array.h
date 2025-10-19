#pragma once

#include <vector>
#include <algorithm>
#include "allocator.h"

//// cpu array for objects, AOS
#define ObjectArray std::vector

//// cpu/gpu array for vector/matrix/real, , SOA
PHYS_NAMESPACE_BEGIN

template<typename T, MemType MT>
class VecArray
{
public:
    VecArray(){
        /**
         * @brief Default constructor for VecArray.
         * 
         * Initializes the size and data members to 0 and nullptr, respectively.
         */
        m_size=0;
        m_data=nullptr;
    }
    
    VecArray(unsigned int size){
        /**
         * @brief Constructor for VecArray with specified size.
         * 
         * Initializes the size member to the specified size and the data member to nullptr.
         * If the size is not zero, it calls the alloc function to allocate memory for the array.
         * 
         * @param size The size of the array.
         */
        m_size=size;
        m_data=nullptr;
        if(m_size!=0) this->alloc(size);
    }
    
    VecArray(unsigned int size, const T& t){
        /**
         * @brief Constructor for VecArray with specified size and value.
         * 
         * Initializes the size member to the specified size and the data member to nullptr.
         * If the size is not zero, it calls the alloc function to allocate memory for the array,
         * and then calls the fill function to fill the array with the specified value.
         * 
         * @param size The size of the array.
         * @param t The value to fill the array with.
         */
        m_size=size;
        m_data=nullptr;
        if(m_size!=0){
            this->alloc(size);
            this->fill(size, t);
        }
    }
    
    int size() const { 
        /**
         * @brief Get the size of the array.
         * 
         * @return The size of the array.
         */
        return m_size; 
    }
    
    void resize(unsigned int size, bool bKeepData=true){
        /**
         * @brief Resize the array.
         * 
         * Allocates a new array of the specified size.
         * If bKeepData is true and the old array is not empty,
         * it copies the data from the old array to the new array.
         * Finally, it frees the old array and updates the size member.
         * 
         * @param size The new size of the array.
         * @param bKeepData Whether to keep the data from the old array.
         */
        T* data;
        allocArray<T,MT>(&data, size);
        if(bKeepData && m_data) copyArray<T,MT,MT> (&data, &m_data, min(size, m_size)); 
        if(m_data) freeArray<T,MT>(&m_data);
        m_data=data;
        m_size=size;
    }

    void release() { if (m_data) freeArray<T, MT>(&m_data); m_data = nullptr; }

    void swap(VecArray<T, MT>& va) {
        if (m_size == 0 || va.m_size == 0) {
            LOG_OSTREAM_WARN << "Swap array: empty array" << std::endl;
            return;
        }
        if (this->m_size != va.size()) {
            LOG_OSTREAM_ERROR << "Swap array: different size" << std::endl;
            return;
        }
        T* tmp = this->m_data;
        this->m_data = va.m_data;
        va.m_data = tmp;
    }

    void fill(unsigned int size, const T& t) { fillArray<T, MT>(&m_data, t, size); }

    void alloc(unsigned int size) {
        allocArray<T, MT>(&m_data, size);
        m_size = size;
    }

    PE_CUDA_FUNC inline unsigned int size() { return m_size; }
    PE_CUDA_FUNC inline T& operator[](unsigned int i) { return m_data[i]; }
    PE_CUDA_FUNC inline T operator[](unsigned int i) const { return m_data[i]; }


protected:
    int m_size; // Size of the array

public:
    T* m_data; // Pointer to the data array
};

/**
 * @brief Macro for defining set and get methods for a structure of arrays (SOA) data member.
 * 
 * @param T The type of the data member.
 * @param MT The memory type of the data member.
 * @param name The name of the data member (lowercase).
 * @param Name The name of the data member (uppercase).
 */
#define DEFINE_SOA_SET_GET(T,MT,name,Name)\
    protected:\
    VecArray<T,MT> name;\
    public:\
    void set##Name(int idx, const T& t){ name[idx] = t; }\
    const T& get##Name(int idx) const { return name[idx]; }\
    VecArray<T,MT>& get##Name##Ref() { return name; }



/**
 * @brief Macro for defining get methods for a structure of arrays (SOA) data member.
 * 
 * @param T The type of the data member.
 * @param MT The memory type of the data member.
 * @param name The name of the data member (lowercase).
 * @param Name The name of the data member (uppercase).
 */
#define DEFINE_SOA_GET(T,MT,name,Name)\
    protected:\
    VecArray<T,MT> name;\
    public:\
    const T& get##Name(int idx) const { return name[idx]; }\
    VecArray<T,MT>& get##Name##Ref() { return name; }


/**
 * @brief Macro for using the set and get methods from a base class for a structure of arrays (SOA) data member.
 * 
 * @param name The name of the data member (lowercase).
 * @param Name The name of the data member (uppercase).
 */
#define USING_BASE_SOA_SET_GET(name,Name)\
    using Base::name; using Base::set##Name; using Base::get##Name; using Base::get##Name##Ref;
/**
 * @brief Macro for using the get methods from a base class for a structure of arrays (SOA) data member.
 * 
 * @param name The name of the data member (lowercase).
 * @param Name The name of the data member (uppercase).
 */
#define USING_BASE_SOA_GET(name,Name)\
    using Base::name; using Base::get##Name; using Base::get##Name##Ref;


/**
 * @brief The maximum number of neighbors.
 */
#define MAX_NEIGHBOR_NUM 126

class NbrList {
public:
    /**
     * @brief Default constructor for NbrList.
     */
    __host__ __device__ NbrList() { cnt = 0; }
    
    // NbrList(const NbrList& nbrlist) {cnt=nbrlist.cnt; nbr_pair=nblist.nbrpair;}
    
    /**
     * @brief Add a neighbor to the NbrList.
     * 
     * @param idx The index of the neighbor.
     * @param dist The distance to the neighbor.
     */
    // __host__ __device__ void add(int idx, Real dist) { if (cnt < MAX_NEIGHBOR_NUM) { nbr_pair[cnt] = make_vec2r(Real(idx), dist); cnt++; } }
    inline __host__ __device__ void add(int idx, Real dist) { nbr_pair[cnt] = make_vec2r(Real(idx), dist); cnt++; }
    
    // __host__ __device__ vec2r get(int off) { return nbr_pair[off]; }
    
    /**
     * @brief Get the neighbor at the specified offset.
     * 
     * @param off The offset of the neighbor.
     * @param idx The index of the neighbor (output).
     * @param dist The distance to the neighbor (output).
     */
    __host__ __device__ void get(int off, int& idx, Real& dist) const { const vec2r& pair = nbr_pair[off]; idx = int(pair.x); dist = pair.y; }
    
    vec2r nbr_pair[MAX_NEIGHBOR_NUM]; /**< The array of neighbor pairs. */
    int cnt; /**< The count of neighbors. */
};



//// Read and write API for vert_and_slot combination

/**
 * @brief Retrieves the vertex and slot combination.
 * 
 * @details This function takes in two unsigned integers, `v` and `s`, and
 * returns their combination as a single unsigned integer.
 * 
 * @param v The vertex value.
 * @param s The slot value.
 * @return The combination of `v` and `s` as a single unsigned integer.
 */
__inline__ __host__ __device__ uint getVertSlot(uint v, uint s){
    // return (v<<5)|(s&0x1F);
    // Shift `v` to the left by 6 bits and combine it with the lower 6 bits of `s`.
    return (v<<6)|(s&0x3F);
}

/**
 * @brief Shifts the given value to the right by 6 bits.
 * 
 * @param vs The value to shift.
 * @return The shifted value.
 */
__inline__ __host__ __device__ uint getVertFromInfo(uint vs){
    // return vs>>5;
    return vs>>6;
}

/**
 * @brief Retrieves the slot value from the given info value.
 * 
 * @details This function takes an unsigned integer `vs` as input and returns the slot value
 * extracted from it. The slot value is obtained by performing a bitwise AND operation between
 * `vs` and a bitmask of 0x3F. The result is an unsigned integer representing the slot value.
 * 
 * @param vs The info value.
 * @return The slot value extracted from the info value.
 */
__inline__ __host__ __device__ uint getSlotFromInfo(uint vs){
    // return vs&0x1F;
    return vs&0x3F;
}


/**
 * @brief The size of each constraint slot.
 */
#define CONSTRAINT_SLOT_SIZE 64

/**
 * @brief Class representing a write slot for constraints.
 */
class ConstraintWriteSlot{
    /**
     * @brief The write buffer for each particle to avoid write conflict on the GPU.
     */
public:
    /**
     * @brief Constructor for the ConstraintWriteSlot class.
     * Initializes all slots with zero vectors.
     */
    __host__ __device__ ConstraintWriteSlot(){
        for(int i=0;i<CONSTRAINT_SLOT_SIZE;i++) slots[i]=make_vec3r(0.0);
    }
    
    vec3r slots[CONSTRAINT_SLOT_SIZE]; /**< The array of slots for the write buffer. */
};

/**
 * @brief Class representing the connected particle (vertex) neighbors for each cloth particle (vertex).
 */
class ClothNbrList {
    /**
     * @brief The connected particle (vertex) neighbors for each cloth particle (vertex).
     */
public:
    int nbrs[16];  ///< The array of neighbor indices.
    Real lens[16]; ///< The array of rest lengths between neighbors.
};

/**
 * @brief Class representing the current status of a triangle, including the positions of its vertices, the center, and the normal.
 */
class TriangleInfo {
    /**
     * @brief Current triangle status, including the positions of triangle vertices, the center, and the normal.
     */
public:
    /**
     * @brief Default constructor for TriangleInfo.
     */
    __host__ __device__ TriangleInfo(){;}

    /**
     * @brief Constructor for TriangleInfo.
     *
     * @param p0 The position of the first vertex.
     * @param p1 The position of the second vertex.
     * @param p2 The position of the third vertex.
     * @param cc The center of the triangle.
     * @param nn The normal of the triangle.
     */
    __host__ __device__ TriangleInfo(const vec3r& p0, const vec3r& p1, const vec3r& p2, const vec3r& cc, const vec3r& nn):c(cc),n(nn),padding(0){
        p[0]=p0;
        p[1]=p1;
        p[2]=p2;
    }
    
    vec3r p[3]; ///< The current position of each triangle vertex.
    vec3r c; ///< The center of the triangle.
    vec3r n; ///< The normal of the triangle (unused).
    Real padding; ///< Padding for alignment.
};


/**
 * @brief Class representing the information of a rest triangle.
 */
class RestTriangleInfo {
    /**
     * @brief Triangle information including the index and slot offset of each triangle vertex and the inverse of the constant reference shape matrix D_m used in deformation gradient.
     */
public:
    uint4 vert_and_slot; ///< The combination of vertex index and its slot offset of each vertex. Each int includes the vertex index [0-27], and the vertex TetraWriteSlot's slot index [27-32]. Read/write through getVertSlot/getVertFromInfo/getSlotFromInfo.
    vec4r inv_rest_mat; ///< The inverse of the constant reference shape matrix D_m used in deformation gradient.
};

/**
 * @brief Class representing the connected particle (vertex) neighbors for deformations.
 */
class DeformNbrList {
public:
    int nbrs[16]; ///< The array of neighbor indices.
    Real lens[16]; ///< The array of rest lengths between neighbors.
};

/**
 * @brief Class representing the information of a tetrahedron.
 */
class TetInfo {
public:
    /**
     * @brief Default constructor for TetInfo.
     */
    __host__ __device__ TetInfo(){;}

    /**
     * @brief Constructor for TetInfo.
     *
     * @param p0 The position of the first vertex.
     * @param p1 The position of the second vertex.
     * @param p2 The position of the third vertex.
     * @param p3 The position of the fourth vertex.
     * @param cc The center of the tetrahedron.
     */
    __host__ __device__ TetInfo(const vec3r& p0, const vec3r& p1, const vec3r& p2, const vec3r& p3, const vec3r& cc):c(cc),padding(0) {
        p[0] = p0;
        p[1] = p1;
        p[2] = p2;
        p[3] = p3;
    }
    
    vec3r p[4]; ///< The positions of the four vertices.
    vec3r c; ///< The center of the tetrahedron.
    Real padding; ///< Padding for alignment.
};

/**
 * @brief Class representing the information of a rest tetrahedron.
 */
class RestTetInfo {
public:
    //// todo:merge vert and vert_slot??
    uint4 vert_and_slot; ///< The 4 vertices, each int includes vertex index [0-27], vertex TetraWriteSlot's slot index [27-32].
    vec3r inv_rest_mat[3]; ///< The mat3 (inverse and transpose of the constant reference shape matrix D_m used in deformation gradient).
    Real vol; ///< The rest volume of the tetrahedron.
    Real padding[2]; ///< Padding for alignment.
};


PHYS_NAMESPACE_END