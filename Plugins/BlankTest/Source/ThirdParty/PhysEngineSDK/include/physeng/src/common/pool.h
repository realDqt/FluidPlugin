#pragma once

#include <stdlib.h>

#include "math/math.h"
#include "common/logger.h"
#include "common/array.h"
#include "common/allocator.h"

PHYS_NAMESPACE_BEGIN

/**
 * @brief Pool class implements a basic object pool. 
 */
template<class T, size_t blockSize>
class Pool{
  public:
    Pool():m_usedNum(0),m_freeNum(0),m_freeNode(nullptr){
        m_alignAllocFunc = alignedAllocC11Std;
        m_alignFreeFunc = alignedFreeC11Std;
    }

    ~Pool(){
        //// destroy all T object
        destroyAllT();

        //// free all blocks
		for(auto block = m_blocks.begin(), blockEnd = m_blocks.end(); block != blockEnd; ++block)
			m_alignFreeFunc(*block, alignof(T));
    }

    /**
     * @brief Create a new T object from the pool.
     * 
     * @param args The arguments to forward to the constructor of T.
     * @return A pointer to the newly created T object, or nullptr if the object could not be created.
     */
    template <class ...Args>
    FORCE_INLINE T* create(Args&&... args) {
        // Allocate memory for the new object
		T* t = allocate();
		return t ? new (t) T(std::forward<Args>(args) ...) : nullptr;
	}

    /**
     * @brief Destroy a T object and return its space to the pool.
     * 
     * @param ptr A pointer to the object to be destroyed.
     */
    FORCE_INLINE void destroy(T* const ptr) {
        // Check if the pointer is not null
		if(ptr) {
            // Call the destructor of the object
			ptr->~T();
            // Deallocate the memory of the object
			deallocate(ptr);
		}
	}

    /**
     * @brief Return the number of free nodes.
     */
    int getFreeNum(){ return m_freeNum; }

    /**
     * @brief Return the number of used nodes
     */
    int getUsedNum(){ return m_usedNum; }
    
    /**
     * @brief Return the occupied space
     */
    int getOccupiedSpace(){ return blockSize * m_blocks.size(); }
    
  protected:
    
    ObjectArray<void*> m_blocks; //!< The memory blocks that the pool uses
    /**
     * @brief FreeNode structure is used to construct the free-node chain.
     */
  	struct FreeNode {
		FreeNode* next;
	};
    FreeNode* m_freeNode; //!< The header of the free-node chain
    int m_usedNum; //!< The number of used node
    int m_freeNum; //!< The number of free node

    void* (*m_alignAllocFunc)(size_t size, size_t align); //!< The memory alloc function
    void (*m_alignFreeFunc)(void* ptr, size_t align); //!< The memory release function
    
    /**
     * @brief Allocate a T object from the pool
     * 
     * @return Pointer to the allocated object
     */
    FORCE_INLINE T* allocate() {
        // If there are no free nodes available, allocate a new block
		if(m_freeNode == nullptr)
			allocBlock();

        // Get the pointer to the free node
		T* ptr = reinterpret_cast<T*>(m_freeNode);

        // Update the free node to the next node in the pool
		m_freeNode = m_freeNode->next;

        // Update the counters
        m_usedNum++;
        m_freeNum--;

        return ptr;
	}
    
    /**
     * @brief Deallocate a T object
     * 
     * @param ptr Pointer to the object to deallocate
     */
    FORCE_INLINE void deallocate(T* ptr) {
        // Check if the pointer is not null
		if(ptr) {
			ASSERT(m_usedNum);
            // Decrement the usedNum counter
            m_usedNum--;

            // Push the pointer to the free node list
			pushFreeNode(reinterpret_cast<FreeNode*>(ptr));
		}
	}

    /**
     * @brief Pushes a free node into the free node chain.
     * 
     * @param ptr Pointer to the free node to be pushed.
     */
    void pushFreeNode(FreeNode* ptr){
        if(ptr){
            // Increment the count of free nodes.
            m_freeNum++;

            // Set the next pointer of the free node to the current head of the free node chain.
            ptr->next = m_freeNode;

            // Set the head of the free node chain to the new free node.
            m_freeNode = ptr;
        }
    }

    /**
     * @brief Allocate a new block when running out of free nodes.
     *
     * @details This function allocates a new block of memory using the aligned allocation function.
     * It then pushes the allocated block into the 'm_blocks' vector.
     * After that, it iterates through the memory block and pushes each element as a free node
     * into the free node list.
     */
    void allocBlock() {
        // Allocate a new block of memory
		T* block = reinterpret_cast<T*>(m_alignAllocFunc(blockSize, alignof(T)));

        // Push the allocated block into the 'm_blocks' vector
		m_blocks.push_back((void*)block);

        // Iterate through the memory block in reverse and push each element as a free node
		T* it = block + (blockSize / sizeof(T));
		while(--it >= block)
			pushFreeNode(reinterpret_cast<FreeNode*>(it));
	}

    /**
     * @brief Release all T objects in destructor
     * 
     * @details This function releases all T objects in the destructor. It iterates over the 
     * linked list of free nodes and pushes them to the freeNodes vector. Then, it 
     * sorts the freeNodes vector and the m_blocks vector in ascending order based 
     * on memory addresses. Finally, it iterates over each block in the m_blocks 
     * vector and destructs the T objects that are not present in the freeNodes 
     * vector.
     */
    void destroyAllT(){
        ObjectArray<T*> freeNodes;
        FreeNode* iter = m_freeNode;

        // Iterate over the linked list of free nodes and push them to the freeNodes vector
        while(iter) {
			freeNodes.push_back(reinterpret_cast<T*> (iter));
			iter = iter->next;
		}
        
		static auto cmpTPtr = [=](T* i, T* j) { return i < j; };
		static auto cmpVPtr = [=](void* i, void* j) { return i < j; };
        // Sort the freeNodes vector in ascending order based on memory addresses
        std::sort(freeNodes.begin(), freeNodes.end(), cmpTPtr); //// low addr -> high addr
        // Sort the m_blocks vector in ascending order based on memory addresses
        std::sort(m_blocks.begin(), m_blocks.end(), cmpVPtr); //// low addr -> high addr

        int j = 0;

        // Iterate over each block in the m_blocks vector
        for(int i = 0; i < m_blocks.size(); i++){
            T* elem = reinterpret_cast<T*> (m_blocks[i]);
            T* bEnd = elem + (blockSize / sizeof(T));

            // Iterate over each T object in the current block
            for(; elem != bEnd; elem++){
                // If the current T object is present in the freeNodes vector, move to the next element
                if(freeNodes[j] == elem)
                    j++;
                // If the current T object is not present in the freeNodes vector, destruct it
                else if(freeNodes[j] > elem){
                    elem->~T();
                }
                // Error case: freeNodes[j] < elem
                else{
                    LOG_OSTREAM_ERROR << "freeNodes[j] < elem" << std::endl;
                }
            }
        }
    }
};

PHYS_NAMESPACE_END
