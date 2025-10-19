// Ensures this header file is included only once in a single compilation, preventing redefinition errors
#pragma once

#include <vector> // Includes the standard vector library for using dynamic arrays

// Class designed for searching and identifying neighboring points in a space
class NeighborSearcher {
public:
    // Constructor declaration
    NeighborSearcher(); 
    // Destructor declaration
    ~NeighborSearcher(); 
    // Method to find and return a list of neighbor identifiers (e.g., indices of points that are considered neighbors)
    std::vector<int> findNeighbors();
    // Method to set or update the point around which neighbors will be searched for.
    // Note: The method signature does not specify parameters, suggesting that this might be an oversight or
    // that the point data is set through some other mechanism (not directly visible in this interface).
    void setPoint();
};
