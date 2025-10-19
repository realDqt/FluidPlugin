// Ensures that this header file is included only once in a single compilation, avoiding duplicate definitions
#pragma once

#include <vector> // Includes the standard vector library for using dynamic arrays
#include "object/gas_world.h" // Includes gas world definitions, which might contain relevant types or utilities

// Class representing the particles involved in splash simulations
class SplashParticles {
public:
    // Dynamic array storing the mass of each particle
    std::vector<real> m_mass; 
    // Dynamic array storing the 3D position of each particle
    std::vector<vec3r> m_position;
    // Dynamic array storing the 3D velocity of each particle
    std::vector<vec3r> m_velocity; 
    // Constructor declaration
    SplashParticles(); 
    // Virtual destructor to ensure derived classes are cleaned up correctly
    virtual ~SplashParticles(); 
    // Initializes the splash particle system, potentially setting up initial states
    void init(); 
    // Spawns new particles and returns the number of particles spawned
    int spawnParticles(); 
};

// Class representing the particles involved in liquid simulations
class LiquidParticles {
public:
    // Dynamic array storing the mass of each liquid particle
    std::vector<real> m_mass; 
    // Dynamic array storing the 3D position of each liquid particle
    std::vector<vec3r> m_position; 
    // Dynamic array storing the 3D velocity of each liquid particle
    std::vector<vec3r> m_velocity; 
    // Constructor declaration
    LiquidParticles(); 
    // Virtual destructor to ensure proper cleanup of derived classes
    virtual ~LiquidParticles(); 
    // Initializes the liquid particle system, setting up initial conditions or states
    void init(); 
    // Function to spawn new liquid particles and return the count of newly spawned particles
    int spawnParticles(); 
};
