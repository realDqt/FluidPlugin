// Ensures this header is included only once during compilation to prevent duplicate definitions
#pragma once

#include <vector> // Includes the vector library for managing dynamic arrays of objects
#include "object/gas_world.h" // Includes the gas world definitions, potentially for atmospheric effects on liquids
#include "Particles.h" // Includes particle definitions for representing liquid and splash particles
#include "Neighbor.h" // Includes neighbor finding utilities, likely for particle-based simulations

// Class for simulating splash effects, focusing on transient, dynamic behaviors
class SplashSimulation {
public:
    // Container for particles specific to splash effects
    SplashParticles m_particles; 
    // Constructor declaration
    SplashSimulation();
    // Virtual destructor for cleanup, ensures compatibility with inheritance
    virtual ~SplashSimulation(); 

    // Simulation methods
    // Moves particles based on their velocities, simulating fluid flow
    void advect(); 
    // Applies gravitational forces to the particles, affecting their motion
    void applyGravity();
    // Initializes particles and properties to simulate a splash effect
    void createSplash(); 
    // Prepares the simulation, setting initial conditions or parameters
    void init(); 
    // Computes and applies forces to particles, such as from interactions or external influences
    void solveForce(); 
};

// Class for simulating liquid behaviors, focusing on more stable and continuous aspects of fluids
class LiquidSimulation {
public:
    // Container for particles that represent the liquid
    LiquidParticles m_particles; 
    // Constructor declaration
    LiquidSimulation(); 
    // Virtual destructor for cleanup, ensures compatibility with inheritance
    virtual ~LiquidSimulation(); 

    // Simulation methods
    // Moves liquid particles based on their velocities, simulating fluid dynamics
    void advect(); 
    // Applies gravity to the liquid particles, influencing their motion
    void applyGravity(); 
    // Calculates interactions between the liquid and its surrounding environment
    void computeEnvironmentCoupling(); 
    // Computes pressure within the liquid based on particle density and applies corresponding forces
    void computePressure(); 
    // Calculates and applies forces between the liquid and solid objects
    void computeRigidCoupling(); 
    // Computes the viscous forces within the liquid, affecting flow characteristics
    void computeViscosity();
    // Computes surface tension effects at the liquid-air interface, influencing shape and stability
    void computeSurfaceTension(); 
    // Enhances the vorticity within the fluid to simulate small-scale swirling effects
    void computeVorticityConfinement(); 
    // Sets up the initial state of the liquid simulation with appropriate particles and properties
    void createLiquid();
    // Initializes the liquid simulation, preparing any necessary data structures or parameters
    void init(); 
};
