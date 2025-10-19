// Include the necessary headers for vector usage, gas world objects, and other related components
#include <vector>
#include "object/gas_world.h"
#include "GasField.h"
#include "GasSimulation.h"
#include "GasRenderer.h"

// Manager class that orchestrates gas simulations and rendering
class GasManager {
public:
    // Holds pointers to GasSimulation objects, managing multiple simulations
    std::vector<GasSimulation*> m_simulations; 
    // An instance of GasRenderer to handle the rendering of gas simulations
    GasRenderer m_renderer; 
    // Constructor declaration
    GasManager();
    // Virtual destructor for polymorphic cleanup
    virtual ~GasManager(); 

    // Methods to create different types of gas sources, initializing their respective simulations
    // Creates a smoke simulation
    void createSmoke(); 
    // Creates a dust simulation
    void createDust(); 
    // Creates a nuclear source simulation
    void createNuclearSource(); 
    // Creates a biochemical gas source simulation
    void createBiochemicalGasSource();
    // Initializes the gas manager, possibly setting up simulations and the renderer
    void init();
    // Renders the current state of gas simulations
    void render(); 
    // Placeholder for a method to retrieve simulation details (not implemented here)
    void getSimulation();
    // Placeholder for a method to retrieve certain attributes of simulations (not implemented here)
    void getAttribute(); 

    // Parameter setting methods for simulations
    // Sets the lifetime parameter of simulations
    void setLifetimeParameter(real l); 
    // Sets the resistance parameter affecting gas movement
    void setResistanceParameter(real r); 
    // Sets the vorticity parameter to adjust swirling motion in the gas
    void setVorticityParameter(real v);
    // Updates the state of all simulations by a time step dt
    void update(real dt); 
};
