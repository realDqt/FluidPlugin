#include "gas_system.h"
#include "nuclear_system.h"
#include "config.h"
#include <vector>
#include "cuda_viewer/cuda_viewer.h"

#define GAS_GRID_SIZE 60
#define NUCLEAR_GRID_SIZE 100

// GasWorld class represents the simulation world for gas and nuclear systems
class GasWorld
{
private:
    std::vector<GasSystem*> gasArray;        // Vector to store pointers to GasSystem objects
    std::vector<NuclearSystem*> nuclearArray; // Vector to store pointers to NuclearSystem objects

    // Macros to define getter and setter functions for class members
    DEFINE_MEMBER_SET_GET(Real, dt, Dt);
    DEFINE_MEMBER_SET_GET(vec3r, origin, Origin);

    // Flag indicating whether gas systems are in use
    bool useGas = true;

public:
    /**
     * @brief Constructor for GasWorld.
     */
    GasWorld();

    /**
     * @brief Destructor for GasWorld.
     */
    ~GasWorld();

    /**
     * @brief Initialize a gas system with specified parameters.
     */
    int initGasSystem(vec3r origin, Real vorticity, Real diffusion, Real buoyancy, Real vcEpsilon, Real decreaseDensity);

    /**
     * @brief Initialize a gas system using parameters from a configuration file.
     */
    int initGasSystem(const char* config_file);

    /**
     * @brief Initialize a nuclear system with specified parameters.
     */
    int initNuclearSystem(vec3r origin, Real vorticity);

    /**
     * @brief Get a pointer to the GasSystem at the specified index.
     */
    GasSystem* getGas(int index);

    /**
     * @brief Get a pointer to the NuclearSystem at the specified index.
     *
     * @param index The index of the NuclearSystem to retrieve.
     * @return A pointer to the NuclearSystem at the specified index.
     */
    NuclearSystem* getNuclear(int index) { return nuclearArray[index]; }

    /**
     * @brief Set rendering data for visualization of a gas system at the specified index.
     */
    void setRenderData(int index, vec3r color, vec3r lightDir, Real decay, unsigned char ambient);

    /**
     * @brief Get the Average Density object
     * 
     * @param index 
     * @return Real 
     */
    Real getAverageDensity(int index);

    /**
     * @brief Initialize a CudaViewer for visualization of a gas system at the specified index.
     */
    void initViewer(int index, CudaViewer& viewer);

    /**
     * @brief Update the CudaViewer for visualization of a gas system at the specified index.
     */
    void updateViewer(int index, CudaViewer& viewer);

    /**
     * @brief Set whether a gas system at the specified index should dissipate.
     */
    void setIfDissipate(int index, bool flag);

    /**
     * @brief Update the simulation for a gas system at the specified index.
     */
    void update(int index);

    /**
     * @brief Set the gas system to simulate fire and smoke.
     */
    void setFireSmoke(int index);

    /**
     * @brief Set the gas system to simulate explosive smoke.
     */
    void setExplodeSmoke(int index);

    /**
     * @brief Set the gas system to simulate exhaust.
     */
    void setExhaust(int index);

    /**
     * @brief Set the gas system to simulate dust.
     */
    void setDust(int index);

    /**
     * @brief Set the gas system to simulate biochemistry effects.
     */
    void setBiochemistry(int index);

    /**
     * @brief Set wind parameters for the gas simulation.
     */
    void setWind(Real strength, vec3r direction);
};

/**
 * @brief Function declaration for initializing a NuclearSystem with vorticity.
 */
NuclearSystem* init(Real vorticity);
