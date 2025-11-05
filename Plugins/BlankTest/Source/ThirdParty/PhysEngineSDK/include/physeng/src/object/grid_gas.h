#pragma once
#include "object/grid_system.h"
#include "common/soa.h"
#include "common/array.h"

PHYS_NAMESPACE_BEGIN

/**
 * @brief Generates smoke particles.
 *
 * @param size the size of the smoke particles
 * @param source the position of the smoke source
 * @param radius the radius of the smoke source
 * @param newDensity the new density of the smoke particles
 * @param density an array of densities for each particle
 * @param newVelocity the new velocity of the smoke particles
 * @param velocity an array of velocities for each particle
 */
void GenerateSmoke(int size, int3 source, int radius, Real newDensity, VecArray<Real, MemType::GPU>& density, vec3r newVelocity, VecArray<vec3r, MemType::GPU> velocity);

/**
 * @brief Applies buoyancy forces to the gas grid.
 *
 * @param size - the size of the gas grid
 * @param temperature - the temperature of the gas grid
 * @param density - the density of the gas grid
 * @param velocity - the velocity of the gas grid
 * @param buoyancyAlpha - the buoyancy alpha parameter
 * @param buoyancyBeta - the buoyancy beta parameter
 * @param gravity - the gravity vector
 */
void AddBuoyancy(int size, Real dt, vec3r buoyancyDirection, VecArray<Real, MemType::GPU>& density, VecArray<vec3r, MemType::GPU>& velocity);

/**
 * @brief Applies wind forces to the gas grid.
 *
 * @param size - the size of the gas grid
 * @param velocity - the velocity of the gas grid
 * @param windDirection - the direction of the wind
 * @param windStrength - the strength of the wind
 */
void AddWind(int size, Real dt, vec3r windDirection, Real windStrength, VecArray<Real, MemType::GPU>& density, VecArray<vec3r, MemType::GPU>& velocity);

/**
 * @brief Locks the rigid particles in place to simulate rigid body motion.
 * 
 * @param size The number of particles.
 * @param density The array of particle densities.
 * @param velocity The array of particle velocities.
 * @param rigidFlag The array of flags indicating whether a particle is rigid or not.
 */
void LockRigid(int size, VecArray<Real, MemType::GPU>& density, VecArray<vec3r, MemType::GPU>& velocity, VecArray<bool, MemType::GPU>& rigidFlag);

/**
 * @brief Applies vorticity confinement to the velocity field.
 *
 * @param size The size of the velocity field.
 * @param dt The time step.
 * @param velocity The velocity field.
 * @param curl The curl field.
 * @param rigidFlag The flag indicating if a cell is rigid.
 */
void VorticityConfinement(int size, Real dt, VecArray<vec3r, MemType::GPU>& velocity, VecArray<vec3r, MemType::GPU>& curl, VecArray<bool, MemType::GPU>& rigidFlag);

/**
 * @brief Diffuses the velocity field over time.
 *
 * @param size The size of the velocity field.
 * @param dt The time step.
 * @param velocity The velocity field.
 * @param tempVelocity The temporary velocity field.
 * @param rigidFlag The flag indicating if a cell is rigid.
 */
void DiffuseVelocity(int size, Real dt, VecArray<vec3r, MemType::GPU>& velocity, VecArray<vec3r, MemType::GPU>& tempVelocity, VecArray<bool, MemType::GPU>& rigidFlag);

/**
 * @brief Updates the project state of the simulation.
 *
 * @param size The size of the simulation grid.
 * @param velocity The velocity field of the simulation.
 * @param divergence The divergence field of the simulation.
 * @param pressure The pressure field of the simulation.
 * @param rigidFlag The flag indicating whether a cell is rigid or not.
 */
void Project(int size, VecArray<vec3r, MemType::GPU>& velocity, VecArray<Real, MemType::GPU>& divergence, VecArray<Real, MemType::GPU>& pressure, VecArray<bool, MemType::GPU>& rigidFlag);

/**
 * @brief Advances the velocity field using an advection scheme.
 *
 * @param size The size of the velocity field.
 * @param dt The time step size.
 * @param velocity The current velocity field.
 * @param tempVelocity The temporary velocity field.
 * @param rigidFlag The flag indicating if a cell is part of a rigid body.
 */
void AdvectVelocity(int size, Real dt, VecArray<vec3r, MemType::GPU>& velocity, VecArray<vec3r, MemType::GPU>& tempVelocity, VecArray<bool, MemType::GPU>& rigidFlag);

/**
 * @brief Diffuses the density values over time.
 * 
 * @param size The size of the density array.
 * @param dt The time step size.
 * @param density The density array to diffuse.
 * @param tempDensity The temporary density array.
 * @param rigidFlag The flag array to indicate rigid bodies.
 */
void DiffuseDensity(int size, Real dt, VecArray<Real, MemType::GPU>& density, VecArray<Real, MemType::GPU>& tempDensity, VecArray<bool, MemType::GPU>& rigidFlag);

/**
 * @brief Advects the density using velocity field over a given time step.
 * 
 * @param size The size of the density grid.
 * @param dt The time step.
 * @param density The density grid to advect.
 * @param tempDensity Temporary density grid used for calculations.
 * @param velocity The velocity field.
 * @param rigidFlag A flag indicating if a cell is part of a rigid body.
 */
void AdvectDensity(int size, Real dt, VecArray<Real, MemType::GPU>& density, VecArray<Real, MemType::GPU>& tempDensity, VecArray<vec3r, MemType::GPU>& velocity, VecArray<bool, MemType::GPU>& rigidFlag);

/**
 * @brief Decreases the density of a given array by a specified disappear rate.
 *
 * @param size The size of the array.
 * @param disappearRate The rate at which the density should decrease.
 * @param density The array of density values to be modified.
 */
void DecreaseDensity(int size, Real disappearRate, VecArray<Real, MemType::GPU>& density);

template<MemType MT>
class GasGrid
{
  public:
    // Getter for m_lengthPerCell
	DEFINE_MEMBER_GET(Real, m_lengthPerCell, LengthPerCell);
	// Setter and getter for m_density
    DEFINE_SOA_SET_GET(Real, MT, m_density, Density);
	// Setter and getter for m_tempDensity
	DEFINE_SOA_SET_GET(Real, MT, m_tempDensity, TempDensity);
	// Setter and getter for m_velocity
	DEFINE_SOA_SET_GET(vec3r, MT, m_velocity, Velocity);
	// Setter and getter for m_tempVelocity
	DEFINE_SOA_SET_GET(vec3r, MT, m_tempVelocity, TempVelocity);
	// Setter and getter for m_pressure
	DEFINE_SOA_SET_GET(Real, MT, m_pressure, Pressure);
	// Setter and getter for m_divergence
	DEFINE_SOA_SET_GET(Real, MT, m_divergence, Divergence);
	// Setter and getter for m_rigidFlag
	DEFINE_SOA_SET_GET(bool, MT, m_rigidFlag, RigidFlag);
  public:
	/**
     * @brief GasGrid constructor.
     *
     * @param gridSize - the size of the grid in each dimension
     * @param lengthPerCell - the length of each cell in the grid
     */
	GasGrid(uint3 gridSize, Real lengthPerCell){
		m_lengthPerCell = lengthPerCell;
		int size = gridSize.x * gridSize.y * gridSize.z;
		if (size != 0) {
			m_density.resize(size);
			m_tempDensity.resize(size);
			m_velocity.resize(size);
			m_tempVelocity.resize(size);
			m_pressure.resize(size);
			m_divergence.resize(size);
			m_rigidFlag.resize(size);
		}
	}

	/**
     * @brief GasGrid destructor.
     */
	virtual ~GasGrid() {
		LOG_OSTREAM_DEBUG<<"release m_density 0x"<<std::hex<<&m_density<<std::dec<<std::endl;
		m_density.release();
		m_tempDensity.release();
		m_velocity.release();
		m_tempVelocity.release();
		m_pressure.release();
		m_divergence.release();
		m_rigidFlag.release();
		LOG_OSTREAM_DEBUG<<"release GridSystem finished"<<std::endl;
	}

	/**
	 * @brief Resizes the grid and initializes the necessary arrays.
	 *
	 * @param gridSize the size of the grid in each dimension
	 * @param lengthPerCell the length of each cell in the grid
	 */
	void resize(uint3 gridSize, Real lengthPerCell) {
		// Set the length per cell
        m_lengthPerCell = lengthPerCell;

		// Calculate the total size of the grid
        int size = gridSize.x * gridSize.y * gridSize.z;

		// Check if the size is not zero
        if (size != 0) {
			// Resize the array
            m_density.resize(size);
            m_tempDensity.resize(size);
            m_velocity.resize(size);
            m_tempVelocity.resize(size);
			m_pressure.resize(size);
			m_divergence.resize(size);
			m_rigidFlag.resize(size);
        }
    };

};

PHYS_NAMESPACE_END