// Ensure this header is only included once in the compilation process
#pragma once

// Include the standard vector library for using dynamic array functionality
#include <vector>
// Include the "gas_world.h" header file from the "object" directory for gas world definitions
#include "object/gas_world.h"

// Base class for a gas field
class GasField {
public:
    // Represents the overall concentration of the gas
    real m_concentration; 
    // Dynamic array holding density values at different points
    std::vector<real> m_density; 
    // Dynamic array holding velocity values at different points
    std::vector<real> m_velocity; 
    // Constructor declaration
    GasField();
    // Virtual destructor for safe polymorphic use
    virtual ~GasField(); 
    // Initialization method declaration
    void init(); 
    // Method to specifically initialize the field
    void initField(); 
};

// Class SmokeField inheriting from GasField, representing a specific type of gas field
class SmokeField : public GasField {
public:
    // Represents the temperature of the smoke
    real m_temperature; 
    // Constructor declaration
    SmokeField(); 
    // Virtual destructor for safe polymorphic use
    virtual ~SmokeField(); 
};

// Class DustField inheriting from GasField, representing a different type of gas field
class DustField : public GasField {
public:
    // Represents the temperature of the dust
    real m_temperature; 
    // Constructor declaration
    DustField(); 
    // Virtual destructor for safe polymorphic use
    virtual ~DustField(); 
};

// Class NuclearField inheriting from GasField, representing a nuclear affected gas field
class NuclearField : public GasField {
public:
    // Represents the temperature within the nuclear field
    real m_temperature; 
    // Constructor declaration
    NuclearField();
    // Virtual destructor for safe polymorphic use
    virtual ~NuclearField(); 
};

// Class BiochemicalGasField inheriting from GasField, representing a biochemical gas field
class BiochemicalGasField : public GasField {
public:
    // Represents the temperature of the biochemical gas
    real m_temperature; 

    // Constructor with an initializer list to set the temperature to 0.0f by default
    BiochemicalGasField() : m_temperature(0.0f);
    // Virtual destructor for safe polymorphic use
    virtual ~BiochemicalGasField(); 
};
