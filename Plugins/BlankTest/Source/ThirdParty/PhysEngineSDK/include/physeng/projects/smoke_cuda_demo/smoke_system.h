#pragma once
#include "math/math.h"
#include <helper_functions.h>
#include "vector_functions.h"
#include "object/grid_fluid.h"

using namespace physeng;
// Particle system class
class SmokeSystem
{
public:
    SmokeSystem(uint3 gridSize, Real cellLength, vec3r worldMin, vec3r worldMax);
    ~SmokeSystem();

    
    //// simulation loop
    void update(Real deltaTime);
    
    //// return particle radius
    Real getCellLength(){ return m_params.cellLength; }

    //// clear    
    void clear();

    //// sample materials
    void addDam(vec3r center, vec3r scale, Real spacing);

    //// submit parameters to device
    void updateParams(){ setGridParameters(&m_params); }

    //// submit host arrays to device
    void submitToDevice(uint start, uint count);

protected: // methods
    SmokeSystem() {}
    void _initialize();
    void _finalize();

public:
    physeng::GridFluid<MemType::GPU> gf = physeng::GridFluid<MemType::GPU>(make_uint3(0, 0, 0), 0);
    GridSimParams m_params; //!< simulation parameters

    DEFINE_SOA_SET_GET(vec3r, MemType::GPU, m_c, Color); //!< particle color array for rendering
    DEFINE_MEMBER_SET_GET(uint, m_solverIterations, SolverIterations); //!< solver iteration
    DEFINE_MEMBER_SET_GET(uint, m_subSteps, SubSteps); //!< solver substeps
    DEFINE_MEMBER_GET(bool, m_isInited, IsInited);

// data
protected: 
    
    // CPU data
    VecArray<Real, MemType::CPU> m_hd;                   //!< grid density
    VecArray<Real, MemType::CPU> m_ht;                   //!< grid temperature
    VecArray<vec3r, MemType::CPU> m_hv;                   //!< particle velocity
    VecArray<vec3r, MemType::CPU> m_hc;                   //!< particle color

    // grid params
    uint3 m_gridSize;
    Real m_cellLength;
    uint m_numCells;

    void resize(int num);
};






