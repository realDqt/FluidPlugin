#pragma once
#include "fluid_system.h"
#include "gas_system.h"
#include "config.h"
#include <vector>
#include "cuda_viewer/cuda_viewer.h"

#define MAX_PARTICLE 200000
#define FLUID_GRID_SIZE 64

// FluidWorld class represents the simulation world for fluid systems
class FluidWorld
{
private:
    std::vector<FluidSystem*> fluidArray; // Vector to store pointers to FluidSystem objects

    // Macros to define getter and setter functions for class members
    DEFINE_MEMBER_SET_GET(Real, dt, Dt);
    DEFINE_MEMBER_SET_GET(vec3r, origin, Origin);
    DEFINE_MEMBER_SET_GET(vec3r, worldMin, WorldMin);
    DEFINE_MEMBER_SET_GET(vec3r, worldMax, WorldMax);

public:
    /**
     * @brief Constructor for FluidWorld that takes world boundaries as input.
     */
    FluidWorld(vec3r worldMin, vec3r worldMax);

    /**
     * @brief Destructor for FluidWorld.
     */
    ~FluidWorld();

    /**
     * @brief Initialize a fluid system with specified parameters.
     */
    int initFluidSystem(vec3r center, vec3r scale, Real kvorticity, Real kviscosity, bool useFoam = false);

    /**
     * @brief Initialize a fluid system using parameters from a configuration file.
     */
    int initFluidSystem(const char* config_file, bool useFoam = false);

    /**
     * @brief Complete initialization for a fluid system at the specified index.
     */
    void completeInit(int index);

    /**
     * @brief Set world boundaries for the simulation.
     */
    void setWorldBoundary(vec3r worldMin, vec3r worldMax);

    /**
     * @brief Get a pointer to the FluidSystem at the specified index.
     */
    FluidSystem* getFluid(int index);

    /**
     * @brief Update a column in the simulation.
     */
    void updateColumn(int idx, Real r, vec3r x);

    /**
     * @brief Add a cube-shaped fluid system to the simulation.
     */
    void addCube(vec3r center, vec3r scale);

    /**
     * @brief Add a sandpile-shaped fluid system to the simulation.
     */
    void addSandpile();

    /**
     * @brief Initialize a CudaViewer for visualization of a fluid system at the specified index.
     */
    void initViewer(int index, CudaViewer& viewer);

    /**
     * @brief Update the CudaViewer for visualization of a fluid system at the specified index.
     */
    void updateViewer(int index, CudaViewer& viewer);

    /**
     * @brief Update the simulation for a fluid system at the specified index.
     */ 
    void update(int index);
};
