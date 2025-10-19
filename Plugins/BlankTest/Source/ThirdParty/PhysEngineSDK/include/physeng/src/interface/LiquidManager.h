// Ensures that the header file is included only once during compilation
#include <vector> // Include the vector library for managing collections of objects
#include "object/fluid_system.h" // Include the fluid system definitions for liquid management
#include "Particles.h" // Include particle system definitions used in simulations
#include "LiquidSimulation.h" // Include the LiquidSimulation class for managing liquid simulations
#include "LiquidRenderer.h" // Include the LiquidRenderer class for rendering liquids

// Class designed to manage all aspects of liquid simulations and their rendering
class LiquidManager {
public:
    // Holds a collection of LiquidSimulation objects for managing multiple liquid simulations
    std::vector<LiquidSimulation> m_liquids; 
    // Holds a collection of SplashSimulation objects for managing multiple splash effects
    std::vector<SplashSimulation> m_splashes; 
    // An instance of LiquidRenderer dedicated to rendering the liquid simulations
    LiquidRenderer m_renderer; 
    // An instance of SplashRenderer dedicated to rendering the splash simulations
    SplashRenderer m_splashRenderer; 
    // Constructor declaration
    LiquidManager(); 
    // Destructor declaration
    ~LiquidManager(); 

    // Method declarations for managing the liquid simulations
    // Initializes and adds a new liquid simulation to m_liquids
    void createLiquid(); 
    // Initializes and adds a new splash simulation to m_splashes
    void createSplash();
    // Initializes the liquid manager, setting up any necessary states before simulations begin
    void init(); 
    // Calls the render methods of m_renderer and m_splashRenderer to draw liquids and splashes
    void render(); 
    // Updates the state of all liquid and splash simulations by a time step dt
    void update(real dt); 
};
