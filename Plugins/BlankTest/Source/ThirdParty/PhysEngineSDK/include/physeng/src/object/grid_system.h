#pragma once

#include <mutex>

#include "common/array.h"

PHYS_NAMESPACE_BEGIN
struct GridSimParams
{
    //// simulation world
    vec3r worldOrigin; //!< origin
    vec3r worldMin;    //!< left lower corner of the world
    vec3r worldMax;    //!< right upper corner of the world

    //// particle grid
    uint3 gridSize; //!< grid size
    uint numCells;       //!< #cells
    Real cellLength;    //!< length of each cell
    Real invCellLength; //!< inverse of length of each cell
    uint3 gridHashMultiplier;

    //// physics parameter
    vec3r gravity;        //!< gravity
    Real kbuoyancy;       //!< buoyancy coefficient
    Real kvorticity;       //!< vorticity coefficient
    Real vc_eps;         //!< epsilon of vorticity confinement
    Real kdiffusion;
    Real bdensity;
    Real btemperature;
    Real disappearRate;
};

extern __constant__ GridSimParams gridParams;
extern GridSimParams hGridParams;
/**
 * Calculates the Euclidean distance of a 3D point from the origin.
 *
 * @param p the 3D point represented as an int3 struct, with x, y, and z coordinates
 *
 * @return the Euclidean distance from the origin, as a Real value
 *
 * @throws None
 */
inline __host__ __device__ Real _d_getDistance(int3 p)
{
    return (Real)sqrt(p.x * p.x + p.y * p.y + p.z * p.z);
}
/**
 * Calculates the square of the distance between the origin and a given point in 3D space.
 *
 * @param p The point in 3D space.
 *
 * @return The square of the distance between the origin and the given point.
 */
inline __host__ __device__ int _d_getDistance2(int3 p)
{
    return p.x * p.x + p.y * p.y + p.z * p.z;
}
/**
 * Returns the grid position of a given vector in 3D space.
 *
 * @param p the input vector whose grid position is to be determined
 *
 * @return the grid position of the input vector as an int3 object
 *
 * @throws None
 */
inline __host__ __device__ int3 _d_getGridPos(vec3r p)
{
    return make_int3(
        floorf(p.x),
        floorf(p.y),
        floorf(p.z));
}
/**
 * Checks if the given grid index is valid.
 *
 * @param gridIdx the grid index to check
 *
 * @return true if the grid index is valid, false otherwise
 */
inline __host__ __device__ bool _d_isValid(int3 gridIdx)
{
    return  (int)gridIdx.x != 0 && (int)gridIdx.x != gridParams.gridSize.x - 1
        && (int)gridIdx.y != 0 && (int)gridIdx.y != gridParams.gridSize.y - 1
        && (int)gridIdx.z != 0 && (int)gridIdx.z != gridParams.gridSize.z - 1;
}
/**
 * Calculates the index of a grid point based on its grid coordinates.
 *
 * @param gridIdx the grid coordinates of the point
 *
 * @return the index of the grid point
 *
 * @throws None
 */
inline __host__ __device__ int _d_getIdx(int3 gridIdx)
{
    return  gridIdx.x + gridIdx.y * gridParams.gridHashMultiplier.y
        + gridIdx.z * gridParams.gridHashMultiplier.z;
}
/**
 * Inline host function that calculates the index based on the given x, y, and z coordinates.
 *
 * @param x the x coordinate
 * @param y the y coordinate
 * @param z the z coordinate
 *
 * @return the calculated index
 */
inline __host__  int _d_getIdx(int x, int y, int z)
{
    return  x + y * gridParams.gridHashMultiplier.y + z * gridParams.gridHashMultiplier.z;
}
/**
 * Gets the grid indices from the given index.
 *
 * @param idx the index
 *
 * @return the grid indices as an int3
 *
 * @throws None
 */
inline __host__ __device__ int3 _d_getGridIdx(int idx)
{
    int gridZ = idx / gridParams.gridHashMultiplier.z;
    int gridY = idx % gridParams.gridHashMultiplier.z / gridParams.gridHashMultiplier.y;
    int gridX = idx % gridParams.gridHashMultiplier.z % gridParams.gridHashMultiplier.y;
    return make_int3(gridX, gridY, gridZ);

}
/**
 * Generates the grid indices for a given index and grid hash multiplier.
 *
 * @param idx The index value.
 * @param gridHashMultiplier The grid hash multiplier.
 *
 * @return The grid indices as int3.
 *
 * @throws None.
 */
inline __host__ __device__ int3 _d_getGridIdx(int idx, uint3 gridHashMultiplier)
{
    int gridZ = idx / gridHashMultiplier.z;
    int gridY = idx % gridHashMultiplier.z / gridHashMultiplier.y;
    int gridX = idx % gridHashMultiplier.z % gridHashMultiplier.y;
    return make_int3(gridX, gridY, gridZ);

}
/**
 * Calculates the interpolated value at a given position in a 3D grid.
 *
 * @param data The data array containing the values of the grid.
 * @param pos The position in 3D space where the value needs to be interpolated.
 *
 * @return The interpolated value at the given position.
 *
 * @throws None
 */
inline __host__ __device__ Real _d_sampleValue(const Real* data, const vec3r pos)
{
    static constexpr int dx[8] = { 0,1,0,1,0,1,0,1 };
    static constexpr int dy[8] = { 0,0,1,1,0,0,1,1 };
    static constexpr int dz[8] = { 0,0,0,0,1,1,1,1 };
    int3 coord = _d_getGridPos(pos);
    vec3r frac = pos - make_vec3r(coord);
    Real w[3][2] = { {1.0 - frac.x,frac.x},{1.0 - frac.y,frac.y} ,{1.0 - frac.z,frac.z} };
    Real intp_value = 0;

    for (int s = 0; s < 8; s++) {
        int d0 = dx[s], d1 = dy[s], d2 = dz[s];
        int idx = _d_getIdx(make_int3(coord.x + d0, coord.y + d1, coord.z + d2));
        intp_value += w[0][d0] * w[1][d1] * w[2][d2] * data[idx];
    }
    return intp_value;
}
/**
 * Calculates the interpolated vector value at a given position in the grid.
 *
 * @param data Pointer to the data array containing the grid values
 * @param pos The position at which to calculate the interpolated value
 *
 * @return The interpolated vector value at the given position
 *
 * @throws None
 */
inline __host__ __device__ vec3r _d_sampleVector(const vec3r* data, const vec3r pos)
{
    static constexpr int dx[8] = { 0,1,0,1,0,1,0,1 };
    static constexpr int dy[8] = { 0,0,1,1,0,0,1,1 };
    static constexpr int dz[8] = { 0,0,0,0,1,1,1,1 };

    int3 coord = _d_getGridPos(pos);
    vec3r frac = pos - make_vec3r(coord);
    Real w[3][2] = { {1.0 - frac.x,frac.x},{1.0 - frac.y,frac.y} ,{1.0 - frac.z,frac.z} };
    vec3r intp_value = make_vec3r(0, 0, 0);

    for (int s = 0; s < 8; s++) {
        int d0 = dx[s], d1 = dy[s], d2 = dz[s];
        int idx = _d_getIdx(make_int3(coord.x + d0, coord.y + d1, coord.z + d2));
        intp_value += w[0][d0] * w[1][d1] * w[2][d2] * data[idx];
    }
    return intp_value;
}
/**
 * Calls the disappear function with the specified number of cells, time step, and density.
 *
 * @param numCells the number of cells
 * @param dt the time step
 * @param density the density of the cells
 */
template<MemType MT>
void callDisappear(int numCells, Real dt, VecArray<Real, MT>& density);
/**
 * Calls the diffusion function for a given number of cells.
 *
 * @param numCells the number of cells to perform diffusion on
 * @param dt the time step size
 * @param tempDensity the temporary density array
 * @param density the density array
 *
 * @throws None
 */
template<MemType MT>
void callDiffuse(int numCells, Real dt, VecArray<Real, MT>& tempDensity, VecArray<Real, MT>& density);

/**
 * Calls the advectProperty function with the specified template parameter.
 *
 * @param numCells the number of cells
 * @param dt the time step
 * @param tempDensity the temporary density array
 * @param density the density array
 * @param velocity the velocity array
 *
 * @throws None
 */
template<MemType MT>
void callAdvectProperty(int numCells, Real dt, VecArray<Real, MT>& tempDensity, VecArray<Real, MT>& density, VecArray<vec3r, MT>& velocity);
/**
 * Calls the advectVelocity function with the given parameters.
 *
 * @param numCells the number of cells
 * @param dt the time step
 * @param tempVelocity the temporary velocity array
 * @param velocity the velocity array
 */
template<MemType MT>
void callAdvectVelocity(int numCells, Real dt, VecArray<vec3r, MT>& tempVelocity, VecArray<vec3r, MT>& velocity);
/**
 * Calls the `addDensity` function with the specified parameters.
 *
 * @param numCells the number of cells
 * @param source the source of the density
 * @param radius the radius of the density
 * @param tempDensity the temporary density value
 * @param density the density vector array
 */
template<MemType MT>
void callAddDensity(int numCells, int3 source, int radius, Real tempDensity, VecArray<Real, MT>& density);
/**
 * Calls the "addBuoyancy" function with the specified parameters.
 *
 * @param numCells the number of cells
 * @param dt the time step
 * @param density the density vector array
 * @param temperature the temperature vector array
 * @param velocity the velocity vector array
 *
 * @throws None
 */
template<MemType MT>
void callAddBuoyancy(int numCells, Real dt, VecArray<Real, MT>& density, VecArray<Real, MT>& temperature, VecArray<vec3r, MT>& velocity);
/**
 * Calls the function `addWind` with the specified parameters.
 *
 * @param numCells the number of cells
 * @param dt the time step
 * @param windForce the force of the wind
 * @param density the array of density values
 * @param temperature the array of temperature values
 * @param velocity the array of velocity vectors
 *
 * @throws None
 */
template<MemType MT>
void callAddWind(int numCells, Real dt, vec3r windForce,  VecArray<Real, MT>& density, VecArray<Real, MT>& temperature, VecArray<vec3r, MT>& velocity);
/**
 * Calls the computeVorticity function on a given number of cells, updating the vorticity and velocity arrays.
 *
 * @param numCells the number of cells to compute vorticity for
 * @param vorticity the array of vorticity vectors to update
 * @param velocity the array of velocity vectors to update
 *
 * @throws None
 */
template<MemType MT>
void callComputeVorticity(int numCells, VecArray<vec3r, MT>& vorticity, VecArray<vec3r, MT>& velocity);
/**
 * Calls the confineVorticity function.
 *
 * @param numCells the number of cells
 * @param dt the time step
 * @param velocity the velocity vector array
 * @param vorticity the vorticity vector array
 *
 * @throws None
 */
template<MemType MT>
void callConfineVorticity(int numCells, Real dt, VecArray<vec3r, MT>& velocity, VecArray<vec3r, MT>& vorticity);
/**
 * Calls the computeDivergence function with the specified parameters.
 *
 * @param numCells the number of cells
 * @param divergence the divergence vector array
 * @param velocity the velocity vector array
 */
template<MemType MT>
void callComputeDivergence(int numCells, VecArray<Real, MT>& divergence, VecArray<vec3r, MT>& velocity);
/**
 * Calls the computePressure function with the given parameters.
 *
 * @param numCells the number of cells
 * @param divergence the divergence vector array
 * @param tempPressure the temporary pressure vector array
 * @param pressure the pressure vector array
 *
 * @throws None
 */
template<MemType MT>
void callComputePressure(int numCells, VecArray<Real, MT>& divergence, VecArray<Real, MT>& tempPressure, VecArray<Real, MT>& pressure);
/**
 * Calls the project function with a given number of cells, velocity, and pressure arrays.
 *
 * @param numCells the number of cells
 * @param velocity the velocity array
 * @param pressure the pressure array
 *
 * @throws None
 */
template<MemType MT>
void callProject(int numCells, VecArray<vec3r, MT>& velocity, VecArray<Real, MT>& pressure);

template<MemType MT>
class GridSystem
{
  public:
	DEFINE_MEMBER_GET(Real, m_lengthPerCell, LengthPerCell);
    DEFINE_SOA_SET_GET(Real, MT, m_divergence, Divergence);
	DEFINE_SOA_SET_GET(Real, MT, m_pressure, Pressure);
	DEFINE_SOA_SET_GET(Real, MT, m_tempPressure, TempPressure);
    DEFINE_SOA_SET_GET(Real, MT, m_density, Density);
	DEFINE_SOA_SET_GET(Real, MT, m_tempDensity, TempDensity);
    DEFINE_SOA_SET_GET(Real, MT, m_temperature, Temperature);
    DEFINE_SOA_SET_GET(Real, MT, m_tempTemperature, TempTemperature);
	DEFINE_SOA_SET_GET(vec3r, MT, m_velocity, Velocity);
	DEFINE_SOA_SET_GET(vec3r, MT, m_tempVelocity, TempVelocity);
    DEFINE_SOA_SET_GET(vec3r, MT, m_vorticity, Vorticity);
  public:
    
	GridSystem(uint3 gridSize, Real lengthPerCell){
		m_lengthPerCell = lengthPerCell;
		int size = gridSize.x * gridSize.y * gridSize.z;
		if (size != 0) {
			m_divergence.resize(size);
			m_pressure.resize(size);
			m_tempPressure.resize(size);
			m_density.resize(size);
			m_tempDensity.resize(size);
            m_temperature.resize(size);
            m_tempTemperature.resize(size);
			m_velocity.resize(size);
			m_tempVelocity.resize(size);
			m_vorticity.resize(size);
		}
	}

	virtual ~GridSystem() {
		LOG_OSTREAM_DEBUG<<"release m_density 0x"<<std::hex<<&m_density<<std::dec<<std::endl;
		m_divergence.release();
		m_pressure.release();
		m_tempPressure.release();
		m_density.release();
		m_tempDensity.release();
        m_temperature.release();
        m_tempTemperature.release();
		m_velocity.release();
		m_tempVelocity.release();
		m_vorticity.release();
		LOG_OSTREAM_DEBUG<<"release GridSystem finished"<<std::endl;
	}

};
void setGridParameters(GridSimParams* hostParams);
PHYS_NAMESPACE_END