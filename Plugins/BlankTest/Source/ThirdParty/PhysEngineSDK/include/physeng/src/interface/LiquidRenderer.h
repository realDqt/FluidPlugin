// Ensures that this header is included only once in the compilation process
#pragma once
#include <vector> // Include the vector library for use with dynamic arrays
#include "object/fluid_system.h" // Include fluid system definitions relevant to liquid rendering
#include "Particles.h" // Include particle system definitions for rendering particles in liquid and splash simulations

// Renderer class for liquids
class LiquidRenderer {
public:
    // Constructor declaration
    LiquidRenderer(); 
    // Virtual destructor for proper cleanup when used in a polymorphic context
    virtual ~LiquidRenderer(); 

    // Rendering methods for different aspects of the liquid
    // Renders the color of the liquid, typically based on its properties like depth or chemical composition
    void drawColor(); 
    // Renders the liquid as a mesh, providing a 3D representation of its surface
    void drawMesh();
    // Renders the normals of the liquid mesh, useful for lighting and shading calculations
    void drawNormal(); 
    // Initializes the renderer, setting up necessary resources or state for liquid rendering
    void init();
    // Performs any cleanup or post-rendering effects after the main render pass
    void postRender(); 
    // Updates the rendering based on changes in the liquid simulation
    void updateLiquid(); 
};

// Renderer class specifically for splash effects
class SplashRenderer {
public:
    // Constructor declaration
    SplashRenderer();
    // Virtual destructor for proper cleanup when used in a polymorphic context
    virtual ~SplashRenderer(); 
    // Initializes the renderer, preparing it for splash rendering
    void init(); 
    // Renders splash effects, visualizing the dynamic and transient nature of splashes
    void renderSplash(); 
};
