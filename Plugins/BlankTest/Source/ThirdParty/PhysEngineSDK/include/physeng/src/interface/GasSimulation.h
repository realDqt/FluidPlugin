// Ensures that this header file is only included once during compilation
#pragma once
#include "object/gas_world.h" // Includes the gas world definitions
#include "GasField.h" // Includes the GasField class for simulation purposes

// Base class for simulating gas dynamics
class GasSimulation {
public:
    // Constructor declaration
    GasSimulation(); 
    // Virtual destructor to ensure derived class objects are properly cleaned up
    virtual ~GasSimulation();

    // Methods for simulating various physical processes within the gas
    // Moves gas particles based on velocity fields to simulate fluid flow
    void advect(); 
    // Calculates buoyancy effects due to temperature differences
    void computeBuoyancy();
    // Computes dissipation of gas properties like velocity and density over time
    void computeDissipation(); 
    // Applies gravity forces to the gas particles
    void computeGravity(); 
    // Manages the lifespan of particles or gas regions
    void computeLifetime();
    // Calculates pressure forces within the gas
    void computePressure();
    // Computes interactions between gas and solid objects
    void computeRigidCoupling(); 
    // Updates the temperature distribution within the gas
    void computeTemperature();
    // Enhances small-scale vortices in the fluid simulation
    void computeVorticityConfinement(); 
    // Applies wind or external forces to the gas
    void computeWindForce();
    // Initializes the simulation with necessary parameters or state
    void init(); 
    // Initializes the gas field specifically for this simulation
    void initField(); 
};

// Derived class for simulating smoke specifically
class SmokeSimulation : public GasSimulation {
    // Holds the smoke-specific gas field characteristics
    SmokeField m_field; 
public:
    // Constructor
    SmokeSimulation();
    // Destructor
    virtual ~SmokeSimulation(); 
    // Initializes smoke-specific parameters or state
    void initSmoke(); 
};

// Derived class for simulating dust
class DustSimulation : public GasSimulation {
    // Holds the dust-specific gas field characteristics
    DustField m_field; 
public:
    // Constructor
    DustSimulation();
    // Destructor
    virtual ~DustSimulation();
    // Initializes dust-specific parameters or state
    void initDust(); 
};

// Derived class for simulating a nuclear affected gas field
class NuclearSimulation : public GasSimulation {
    // Holds the nuclear-specific gas field characteristics
    NuclearField m_field; 
public:
    // Constructor
    NuclearSimulation(); 
    // Destructor
    virtual ~NuclearSimulation();
    // Initializes nuclear-specific parameters or state
    void initSource(); 
};

// Derived class for simulating biochemical gas
class BiochemicalGasSimulation : public GasSimulation {
    // Holds the biochemical-specific gas field characteristics
    BiochemicalGasField m_field; 
public:
    // Constructor
    BiochemicalGasSimulation(); 
    // Destructor
    virtual ~BiochemicalGasSimulation();
    // Initializes biochemical-specific parameters or state
    void initSource(); 
};
