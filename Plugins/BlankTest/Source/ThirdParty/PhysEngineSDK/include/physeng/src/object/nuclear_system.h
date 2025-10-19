#pragma once
#include "math/math.h"
#include <helper_functions.h>
#include "vector_functions.h"
#include "object/grid_nuclear.h"

using namespace physeng;
// Particle system class

class NuclearSystem
{
public:
    NuclearSystem(uint3 gridSize, Real cellLength, vec3r worldMin, vec3r worldMax);
    ~NuclearSystem();

    /**
     * @brief simulation loop
     */
    void update(Real deltaTime);
    
    /**
     * @brief Get the length of a cell in the nuclear system.
     *
     * @return The length of a cell in the nuclear system.
     */
    Real getCellLength(){ return m_params.cellLength; }

    /**
     * @brief Clear.
     */
    void clear();

    /**
     * @brief Sample materials.
     */
    void addDam(vec3r center, vec3r scale, Real spacing);

    /**
     * @brief Submit parameters to device.
     */
    void updateParams(){ setGridParameters(&m_params); }

    /**
     * @brief Submit host arrays to device.
     */
    void submitToDevice(uint start, uint count);

    // void setM_hc(int idx,vec3r color);

protected: 
    // methods
    NuclearSystem() {}
    void _initialize();
    void _finalize();

public:
    physeng::Nuclear<MemType::GPU> gf = physeng::Nuclear<MemType::GPU>(make_uint3(0, 0, 0), 0);
    // simulation parameters
    GridSimParams m_params; 

     // particle color array for rendering
    DEFINE_SOA_SET_GET(vec3r, MemType::GPU, m_c, Color);
     // solver iteration
    DEFINE_MEMBER_SET_GET(uint, m_solverIterations, SolverIterations);
     // solver substeps
    DEFINE_MEMBER_SET_GET(uint, m_subSteps, SubSteps);
    DEFINE_MEMBER_GET(bool, m_isInited, IsInited);
    DEFINE_MEMBER_SET_GET(Real, norm_wind, NormWind);
    DEFINE_SOA_SET_GET(Source, MemType::GPU, sources_G, Source_G);

    bool start_pollution;
    VecArray<Real, MemType::CPU>& getDensity() { return m_hd; }
// data
protected: 
    
    // CPU data
    // grid density
    VecArray<Real, MemType::CPU> m_hd;            
    // grid temperature       
    VecArray<Real, MemType::CPU> m_ht;   
    // particle velocity                
    VecArray<vec3r, MemType::CPU> m_hv;      
    // particle color             
    VecArray<vec3r, MemType::CPU> m_hc;                   

    // grid params
    uint3 m_gridSize;
    Real m_cellLength;
    uint m_numCells;
    Real time_counter;
    //VecArray<Source,MemType::CPU> sources;
    std::vector<Source> sources;
    float3 wind_velocity;
    //float norm_wind;
    float wind_angle;
    // release interval
    Real r_interval;
    float3 Vs;
    VecArray<Real,MemType::GPU> height;    
    // for update velocity
	VecArray<Real,MemType::CPU> temp_height;

    void resize(int num);
};






