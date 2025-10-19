// Ensures that the header file is included only once during compilation
#pragma once

// Includes the vector library for managing collections and the gas world definitions
#include <vector>
#include "object/gas_world.h"
#include "GasField.h" // Includes the GasField class for potential use in rendering

// Renderer class dedicated to visualizing gas fields
class GasRenderer {
public:
    // Constructor declaration
    GasRenderer(); 
    // Virtual destructor to enable subclassing with proper cleanup
    virtual ~GasRenderer(); 
    // Method to draw or render the current state of gas fields to the screen or a visualization context
    void draw(); 
    // Initializes the renderer, setting up necessary resources or state before rendering begins
    void init(); 
    // Updates the renderer's understanding or representation of the gas field, potentially recalculating visual aspects
    void updateField(); 
};
