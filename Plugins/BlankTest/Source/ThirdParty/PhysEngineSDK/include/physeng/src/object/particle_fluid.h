#pragma once
#include "object/particle_system.h"
#include "object/particle_system_util.h"
#include "common/soa.h"

PHYS_NAMESPACE_BEGIN

/**
 * @brief Init particle fluid.
 */
template<MemType MT>
void initParticleFluid();

// template<MemType MT>
// void callAdvect(int size, Real dt, VecArray<vec3r,MT>& vx,  VecArray<vec3r,MT>& vv);

// template<MemType MT>
// void callStackParticle(int size, int x, int y, int z, Real gap, VecArray<vec3r,MT>& vx);

// template<MemType MT>
// void callComputeAttractForce(int size, vec3r center, Real scale, VecArray<vec3r,MT>& vf, VecArray<vec3r,MT>& vx);




void reorderDataAndFindCellStart(uint size, uint* cellStart, uint* cellEnd, vec4r* sortedPositionPhase, vec4r* sortedVel, Real* sortedMass, uint* sortedPhase, uint* gridParticleHash, uint* O2SParticleIndex, uint* S2OparticleIndex, vec3r* oldPos, vec4r* oldVel, Real* oldMass, uint* oldPhase, uint numCells, uint numParticles);

// template<MemType MT>
// void callEnforceFrictionBoundary(int size, VecArray<vec3r,MT>& newPos, VecArray<vec3r,MT>& oldPos, VecArray<vec3r,MT>& sortedPos, VecArray<uint,MT>& sortedPhase, VecArray<uint,MT>& S2OParticleIndex);

/**
 * @brief Resolves penetration between particles in a simulation.
 *
 * @tparam MT The memory type used by the arrays.
 * @param size The number of particles in the simulation.
 * @param tempPosition The temporary array storing the positions of the particles.
 * @param sortedPositionPhase The array storing the sorted positions and phases of the particles.
 * @param sortedMass The array storing the sorted masses of the particles.
 * @param S2OParticleIndex The array storing the indices of the particles.
 * @param cellStart The array storing the start indices of each cell.
 * @param cellEnd The array storing the end indices of each cell.
 * @param rigidParticleSign The array storing the signs of the rigid particles.
 */
template<MemType MT>
void callResolvePenetration(int size, VecArray<vec3r,MT>& tempPosition, VecArray<vec4r,MT>& sortedPositionPhase, VecArray<Real,MT>& sortedMass, VecArray<uint,MT>& S2OParticleIndex, VecArray<uint,MT>& cellStart, VecArray<uint,MT>& cellEnd, VecArray<uint,MT>& rigidParticleSign);

/**
 * @brief Resolves friction for a given set of particles.
 *
 * @tparam MT The memory type.
 * @param size The number of particles.
 * @param tempStarPositionPhase The temporary array to store the star position and phase.
 * @param tempPosition The temporary array to store the position.
 * @param oldPosition The array of old positions.
 * @param sortedPositionPhase The array of sorted positions and phases.
 * @param sortedMass The array of sorted masses.
 * @param S2OParticleIndex The array of S2O particle indices.
 * @param cellStart The array of cell start indices.
 * @param cellEnd The array of cell end indices.
 * @param rigidParticleSign The array of rigid particle signs.
 */
template<MemType MT>
void callResolveFriction(int size, VecArray<vec4r,MT>& tempStarPositionPhase, VecArray<vec3r,MT>& tempPosition, VecArray<vec3r,MT>& oldPosition, VecArray<vec4r,MT>& sortedPositionPhase, VecArray<Real,MT>& sortedMass, VecArray<uint,MT>& S2OParticleIndex, VecArray<uint,MT>& cellStart, VecArray<uint,MT>& cellEnd, VecArray<uint,MT>& rigidParticleSign);

/**
 * @brief Update the lambda values for each particle.
 *
 * @tparam MT The memory type of the arrays.
 * @param size The number of particles.
 * @param lambda The array of lambda values.
 * @param InvDensity The array of inverse density values.
 * @param sortedPositionPhase The array of sorted position and phase values.
 * @param sortedMass The array of sorted mass values.
 * @param S2OParticleIndex The array of S2O particle indices.
 * @param cellStart The array of cell start indices.
 * @param cellEnd The array of cell end indices.
 */
template<MemType MT>
void callUpdateLambda(int size, VecArray<Real,MT>& lambda, VecArray<Real,MT>& InvDensity, VecArray<vec4r,MT>& sortedPositionPhase, VecArray<Real,MT>& sortedMass, VecArray<uint,MT>& S2OParticleIndex, VecArray<uint,MT>& cellStart, VecArray<uint,MT>& cellEnd);

/**
 * @brief Calls the solveFluid function.
 *
 * @tparam MT The type of memory to use.
 * @param size The size of the arrays.
 * @param normal The array of normal vectors.
 * @param newPositionPhase The array of new position and phase vectors.
 * @param lambda The array of lambda values.
 * @param InvDensity The array of inverse density values.
 * @param sortedPositionPhase The array of sorted position and phase vectors.
 * @param sortedMass The array of sorted mass values.
 * @param S2OParticleIndex The array of S2O particle indices.
 * @param cellStart The array of cell start indices.
 * @param cellEnd The array of cell end indices.
 */
template<MemType MT>
void callSolveFluid(int size, VecArray<vec4r,MT>& normal, VecArray<vec4r,MT>& newPositionPhase, VecArray<Real,MT>& lambda, VecArray<Real,MT>& InvDensity, VecArray<vec4r,MT>& sortedPositionPhase, VecArray<Real,MT>& sortedMass, VecArray<uint,MT>& S2OParticleIndex, VecArray<uint,MT>& cellStart, VecArray<uint,MT>& cellEnd);

/**
 * @brief Calls the solveFluidAndViscosity function with the given parameters.
 * 
 * @tparam MT The memory type.
 * @param size The size of the arrays.
 * @param normal The array of normal vectors.
 * @param dv The array of velocity vectors.
 * @param newPositionPhase The array of new position and phase vectors.
 * @param lambda The array of lambda values.
 * @param invDensity The array of inverse density values.
 * @param sortedPositionPhase The array of sorted position and phase vectors.
 * @param sortedVel The array of sorted velocity vectors.
 * @param sortedMass The array of sorted mass values.
 * @param S2OParticleIndex The array of S2O particle indices.
 * @param cellStart The array of cell start indices.
 * @param cellEnd The array of cell end indices.
 */
template<MemType MT>
void callSolveFluidAndViscosity(int size, VecArray<vec4r,MT>& normal, VecArray<vec4r,MT>& dv, VecArray<vec4r,MT>& newPositionPhase, VecArray<Real,MT>& lambda, VecArray<Real,MT>& InvDensityy, VecArray<vec4r,MT>& sortedPositionPhase, VecArray<vec4r,MT>& sortedVel, VecArray<Real,MT>& sortedMass, VecArray<uint,MT>& S2OParticleIndex, VecArray<uint,MT>& cellStart, VecArray<uint,MT>& cellEnd);

// template<MemType MT>
// void callSolveFluidAndViscosity(int size, VecArray<vec3r,MT>& normal, VecArray<vec3r,MT>& dv, VecArray<vec3r,MT>& newPosStar, VecArray<Real,MT>& lambda, VecArray<Real,MT>& InvDensity, VecArray<vec4r,MT>& sortedPositionPhase, VecArray<vec3r,MT>& sortedVel, VecArray<Real,MT>& sortedMass, VecArray<uint,MT>& S2OParticleIndex, VecArray<uint,MT>& cellStart, VecArray<uint,MT>& cellEnd);

/**
 * @brief Updates the surface tension forces and normals for a given size.
 * 
 * @tparam MT The memory type.
 * @param size The size of the arrays.
 * @param force The array of forces.
 * @param normal The array of normals.
 * @param InvDensity The array of inverse densities.
 * @param sortedPositionPhase The array of sorted position phases.
 * @param sortedMass The array of sorted masses.
 * @param S2OParticleIndex The array of S2O particle indices.
 * @param cellStart The array of cell start indices.
 * @param cellEnd The array of cell end indices.
 */
template<MemType MT>
void callUpdateSurfaceTension(int size, VecArray<vec4r,MT>& force, VecArray<vec4r,MT>& normal, VecArray<Real,MT>& InvDensity, VecArray<vec4r,MT>& sortedPositionPhase, VecArray<Real,MT>& sortedMass, VecArray<uint,MT>& S2OParticleIndex, VecArray<uint,MT>& cellStart, VecArray<uint,MT>& cellEnd);

/**
 * @brief Updates the position, velocity, and forces of particles in a simulation.
 * 
 * @tparam MT The memory type of the arrays.
 * @param size The number of particles.
 * @param dt The time step.
 * @param sortedPositionPhase The sorted position phase array.
 * @param oldPos The array of old positions.
 * @param vel The array of velocities.
 * @param dvel The array of velocity changes.
 * @param force The array of forces.
 * @param S2OParticleIndex The array of S2O particle indices.
 * @param cellStart The array of cell start indices.
 * @param cellEnd The array of cell end indices.
 */
template<MemType MT>
void callUpdatePositionVelocity(int size, Real dt, VecArray<vec4r,MT>& sortedPositionPhase, VecArray<vec3r,MT>& oldPos, VecArray<vec4r,MT>& vel, VecArray<vec4r,MT>& dvel, VecArray<vec4r,MT>& force, VecArray<uint,MT>& S2OParticleIndex, VecArray<uint,MT>& cellStart, VecArray<uint,MT>& cellEnd);

/**
 * @brief Calls the solveShapeMatching function with the given parameters.
 * 
 * @tparam MT The memory type.
 * @param size The size of the arrays.
 * @param rigidConstraintCount The number of rigid constraints.
 * @param tempStarPositionPhase The array of temporary star position and phase vectors.
 * @param sortedPositionPhase The array of sorted position and phase vectors.
 * @param constraintStartIndex The array of constraint start indices.
 * @param constraintParticleCount The array of constraint particle counts.
 * @param r The array of r vectors.
 * @param constraintParticleMap The array of constraint particle maps.
 * @param q The array of q vectors.
 * @param mass The array of mass values.
 * @param O2SParticleIndex The array of O2S particle indices.
 * @param particleIndex The array of particle indices.
 */
template<MemType MT>
void callSolveShapeMatching(int size, uint rigidConstraintCount, VecArray<vec4r,MT>& tempStarPositionPhase, VecArray<vec4r,MT>& sortedPositionPhase, VecArray<uint,MT>& constraintStartIndex, VecArray<uint,MT>& constraintParticleCount, VecArray<vec3r,MT>& r, VecArray<uint,MT>& constraintParticleMap, VecArray<vec3r,MT>& q, VecArray<Real,MT>& mass, VecArray<uint,MT>& O2SParticleIndex, VecArray<uint,MT>& particleIndex);

/**
 * @brief Update the vorticity.
 * 
 * @tparam MT The memory type.
 * @param[in] size The size of the arrays.
 * @param[in,out] positions The positions array.
 * @param[in] velocities The velocities array.
 * @param[in,out] vorticity The vorticity array.
 * @param[in,out] forces The forces array.
 * @param[in] masses The masses array.
 * @param[in] neighbors The neighbors array.
 * @param[in] startIndices The start indices array.
 * @param[in] cellIndices The cell indices array.
 */
template<MemType MT>
void callUpdateVorticity(int, VecArray<vec4r,MT>&, VecArray<Real,MT>&, VecArray<vec4r,MT>&, VecArray<vec4r,MT>&, VecArray<Real,MT>&, VecArray<uint,MT>&, VecArray<uint,MT>&, VecArray<uint,MT>&, VecArray<uint,MT>&);

/**
 * @brief Apply the vorticity to the given arrays.
 *
 * @tparam MT - The memory type.
 * @param size - The size of the arrays.
 * @param dt - The time step.
 * @param positions - The positions array.
 * @param velocities - The velocities array.
 * @param vorticity - The vorticity array.
 * @param forces - The forces array.
 * @param densities - The densities array.
 * @param indices - The indices array.
 * @param cellStarts - The cell starts array.
 * @param cellEnds - The cell ends array.
 */
template<MemType MT>
void callApplyVorticity(int, Real, VecArray<vec4r,MT>&, VecArray<vec4r,MT>&, VecArray<Real,MT>&, VecArray<vec4r,MT>&, VecArray<Real,MT>&, VecArray<uint,MT>&, VecArray<uint,MT>&, VecArray<uint,MT>&, VecArray<uint,MT>&);

/**
 * @brief Calls the function to update the neighbor list.
 * 
 * @tparam MT The memory type.
 * @param size The size of the neighbor list.
 * @param nbrLists The neighbor list.
 * @param sortedPositionPhase The sorted position phase.
 * @param S2OParticleIndex The S2O particle index.
 * @param cellStart The cell start.
 * @param cellEnd The cell end.
 */
template<MemType MT>
void callUpdateNbrList(int size, VecArray<NbrList,MT>& nbrLists, VecArray<vec4r,MT>& sortedPositionPhase,VecArray<uint,MT>& S2OParticleIndex, VecArray<uint,MT>& cellStart, VecArray<uint,MT>& cellEnd);

/**
 * @brief Updates lambda values and inverse density using a fast algorithm.
 * 
 * @tparam MT The memory type.
 * @param size The size of the arrays.
 * @param lambda The lambda array to be updated.
 * @param invDensity The inverse density array to be updated.
 * @param nbrLists The neighbor lists array.
 * @param sortedPositionPhase The sorted position and phase array.
 */
template<MemType MT>
void callUpdateLambdaFast(int size, VecArray<Real,MT>& lambda, VecArray<Real,MT>& invDensity, VecArray<NbrList,MT>& nbrLists, VecArray<vec4r,MT>& sortedPositionPhase);

/**
 * @brief Solves fluid fast.
 * 
 * @tparam MT MemType template parameter
 * @param size The size parameter
 * @param normal The normal parameter
 * @param newPositionPhase The newPositionPhase parameter
 * @param lambda The lambda parameter
 * @param invDensity The invDensity parameter
 * @param nbrLists The nbrLists parameter
 * @param sortedPositionPhase The sortedPositionPhase parameter
 */
template<MemType MT>
void callSolveFluidFast(int size, VecArray<vec4r,MT>& normal, VecArray<vec4r,MT>& newPositionPhase, VecArray<Real,MT>& lambda, VecArray<Real,MT>& invDensity, VecArray<NbrList,MT>& nbrLists, VecArray<vec4r,MT>& sortedPositionPhase);

/**
 * @brief Updates the color field based on size, colorField, invDensity, sortedPositionPhase, cellStart, and cellEnd.
 * 
 * @tparam MT The memory type.
 * @param size The size of the arrays.
 * @param colorField The color field array.
 * @param invDensity The inverse density array.
 * @param sortedPositionPhase The sorted position and phase array.
 * @param cellStart The cell start array.
 * @param cellEnd The cell end array.
 */
template<MemType MT>
void callUpdateColorField(int size, VecArray<Real,MT>& colorField, VecArray<Real, MT>& invDensity, VecArray<vec4r,MT>& sortedPositionPhase, VecArray<uint,MT>& cellStart, VecArray<uint,MT>& cellEnd);

/**
 * @brief Updates the normal, colorField, sortedPositionPhase, cellStart, and cellEnd arrays.
 * 
 * @tparam MT The memory type.
 * @param size The size of the arrays.
 * @param normal The normal array.
 * @param colorField The colorField array.
 * @param sortedPositionPhase The sortedPositionPhase array.
 * @param cellStart The cellStart array.
 * @param cellEnd The cellEnd array.
 */
template<MemType MT>
void callUpdateNormal(int size, VecArray<vec4r,MT>& normal, VecArray<Real, MT>& colorField, VecArray<vec4r,MT>& sortedPositionPhase, VecArray<uint,MT>& cellStart, VecArray<uint,MT>& cellEnd);
// void setParameters(SimParams* hostParams);

/**
 * @brief Generates foam particles.
 * 
 * @tparam MT The memory type.
 * @param size The size of the particles.
 * @param rigidbodySize The size of the rigid bodies.
 * @param dt The time step.
 * @param foamParticleCount The count of foam particles.
 * @param foamParticlePositionPhase The position and phase of foam particles.
 * @param foamParticleVelocity The velocity of foam particles.
 * @param foamParticleLifetime The lifetime of foam particles.
 * @param sortedPositionPhase The sorted position and phase.
 * @param normal The normal vector.
 * @param vel The velocity vector.
 * @param S2OParticleIndex The index of the S2O particle.
 * @param cellStart The start index of the cells.
 * @param cellEnd The end index of the cells.
 * @param color The color of the particles.
 */
template<MemType MT>
void generateFoamParticle(int size, int rigidbodySize, Real dt, int* foamParticleCount, VecArray<vec3r, MT> &foamParticlePositionPhase, VecArray<vec4r, MT> &foamParticleVelocity, VecArray<Real, MT> &foamParticleLifetime, VecArray<vec4r,MT>& sortedPositionPhase, VecArray<vec4r,MT>& normal, VecArray<vec4r,MT>& vel, VecArray<uint,MT>& S2OParticleIndex, VecArray<uint,MT>& cellStart, VecArray<uint,MT>& cellEnd, VecArray<vec3r, MT>& color);

/**
 * @brief Advects foam particles based on their position, velocity, and lifetime.
 * 
 * @tparam MT The memory type for the arrays.
 * @param size The number of foam particles.
 * @param dt The time step for advection.
 * @param foamParticlePositionPhase Array of foam particle positions and phases.
 * @param foamParticleVelocity Array of foam particle velocities.
 * @param foamParticleLifetime Array of foam particle lifetimes.
 * @param sortedPositionPhase Array of sorted positions and phases.
 * @param vel Array of velocities.
 * @param S2OParticleIndex Array of S2O particle indices.
 * @param cellStart Array of cell start indices.
 * @param cellEnd Array of cell end indices.
 */
template<MemType MT>
void advectFoamParticle(int size, Real dt, VecArray<vec3r, MT>& foamParticlePositionPhase, VecArray<vec4r, MT>& foamParticleVelocity, VecArray<Real, MT>& foamParticleLifetime, VecArray<vec4r, MT>& sortedPositionPhase, VecArray<vec4r, MT>& vel, VecArray<uint, MT>& S2OParticleIndex, VecArray<uint, MT>& cellStart, VecArray<uint, MT>& cellEnd);

/**
 * @brief Sorts the foam particles based on their index.
 *
 * @tparam MT The memory type.
 * @param size The size of the foam particle array.
 * @param S20FoamParticleIndex The array of foam particle indices.
 * @param foamParticlePositionPhase The array of foam particle positions and phases.
 * @param tempFoamParticlePositionPhase The temporary array of foam particle positions and phases.
 * @param foamParticleVelocity The array of foam particle velocities.
 * @param tempFoamParticleVelocity The temporary array of foam particle velocities.
 * @param foamParticleLifetime The array of foam particle lifetimes.
 */
template<MemType MT>
void sortFoamParticle(int size, VecArray<uint, MemType::GPU> S20FoamParticleIndex, VecArray<vec3r, MT>& foamParticlePositionPhase, VecArray<vec3r, MT>& tempFoamParticlePositionPhase, VecArray<vec4r, MT>& foamParticleVelocity, VecArray<vec4r, MT>& tempFoamParticleVelocity, VecArray<Real, MT>& foamParticleLifetime);

/**
 * @brief Removes a foam particle from the arrays.
 *
 * @tparam MT The type of memory to use for the arrays.
 * @param size The size of the arrays.
 * @param foamParticleCount The array storing the count of foam particles.
 * @param foamParticlePositionPhase The array storing the positions and phases of foam particles.
 * @param foamParticleVelocity The array storing the velocities of foam particles.
 * @param foamParticleLifetime The array storing the lifetimes of foam particles.
 */
template<MemType MT>
void removeFoamParticle(int size, int *foamParticleCount, VecArray<vec3r, MT>& foamParticlePositionPhase, VecArray<vec4r, MT>& foamParticleVelocity, VecArray<Real, MT>& foamParticleLifetime);

/**
 * @brief Initializes the useless particle with the given size.
 * 
 * @tparam MT The memory type.
 * @param size The size of the useless particle.
 * @param foamParticleCount The pointer to the foam particle count.
 * @param foamParticlePositionPhase The foam particle position phase.
 * @param foamParticleVelocity The foam particle velocity.
 * @param foamParticleLifetime The foam particle lifetime.
 */
template<MemType MT>
void initUselessParticle(int size, int *foamParticleCount, VecArray<vec3r, MT>& foamParticlePositionPhase, VecArray<vec4r, MT>& foamParticleVelocity, VecArray<Real, MT>& foamParticleLifetime);

/**
 * @brief Calls the CollideTerrain function with the given parameters.
 *
 * @param size The size of the sortedPositionPhase array.
 * @param sortedPositionPhase The array of sorted 4D vectors representing position and phase.
 * @param terrainHeight The array of terrain heights.
 * @param originPos The origin position.
 * @param edgeCellNum The number of edge cells.
 * @param cellSize The size of each cell.
 */
template<MemType MT>
void callCollideTerrain(int size, VecArray<vec4r, MT>& sortedPositionPhase, VecArray<Real, MT>& terrainHeight, vec3r originPos, uint edgeCellNum, Real cellSize);

/**
 * @brief Calls the collideTerrainFoam function.
 *
 * @param size The size parameter.
 * @param foamParticlePosition The foam particle positions.
 * @param terrainHeight The terrain heights.
 * @param originPos The origin position.
 * @param edgeCellNum The edge cell number.
 * @param cellSize The cell size.
 */
template<MemType MT>
void callCollideTerrainFoam(int size, VecArray<vec3r, MT>& foamParticlePosition, VecArray<Real, MT>& terrainHeight, vec3r originPos, uint edgeCellNum, Real cellSize);

void checkConstraint(int,VecArray<uint,MemType::GPU>&,VecArray<uint,MemType::GPU>&);

template<MemType MT>
class ParticleFluid : public ParticleSystem<MT>{
public:
    using Base=ParticleSystem<MT>;

    //// inherent variables
    USING_BASE_SOA_SET_GET(m_x, Position);                  //!< Base::position
    USING_BASE_SOA_SET_GET(m_tx, TempPosition);             //!< Base::temp position
    USING_BASE_SOA_SET_GET(m_v, Velocity);                  //!< Base::velocity
    USING_BASE_SOA_SET_GET(m_tv, TempVelocity);             //!< Base::temp velocity
    USING_BASE_SOA_SET_GET(m_m, Mass);                      //!< Base::mass

    DEFINE_SOA_SET_GET(vec4r, MT, m_txph, TempPositionPhase);          //!< Base::temp position
    DEFINE_SOA_SET_GET(vec4r, MT, m_tsxph, TempStarPositionPhase);          //!< Base::temp position

    //// new variables for particle pbd
    DEFINE_SOA_SET_GET(uint, MT, m_ph, Phase);              //!< material phase
    DEFINE_SOA_SET_GET(uint, MT, m_h, ParticleHash);        //!< particle hash for space sort
    DEFINE_SOA_SET_GET(uint, MT, m_rs, RigidParticleSign);  //!< rigid body sign

    //// index mapping
    DEFINE_SOA_SET_GET(uint, MT, m_s2o, S2OParticleIndex);  //!< map sorted index to origin index
    DEFINE_SOA_SET_GET(uint, MT, m_o2s, O2SParticleIndex);  //!< map origin index to sorted index

    //// sorted variables
    DEFINE_SOA_SET_GET(Real, MT, m_sm, SortedMass);         //!< sorted mass
    // DEFINE_SOA_SET_GET(vec4r, MT, m_sx, SortedPosition);    //!< sorted position
    DEFINE_SOA_SET_GET(vec4r, MT, m_sv, SortedVelocity);    //!< sorted velocity
    DEFINE_SOA_SET_GET(uint, MT, m_sph, SortedPhase);       //!< sorted phase
    DEFINE_SOA_SET_GET(vec4r, MT, m_sxph, SortedPositionPhase);    //!< sorted position and phase
    DEFINE_SOA_SET_GET(vec4r, MT, m_dv, DeltaVelocity);    //!< delta velocity

    //// intermediate variables
    // DEFINE_SOA_SET_GET(Real, MT, m_rho, Density);           //!< fluid density
    DEFINE_SOA_SET_GET(Real, MT, m_irho, InvDensity);           //!< fluid density inverse
    DEFINE_SOA_SET_GET(Real, MT, m_cf, ColorField);         //!< color field
    DEFINE_SOA_SET_GET(Real, MT, m_l, Lambda);              //!< lambda in fluid density constraint
    DEFINE_SOA_SET_GET(vec4r, MT, m_vort, Vorticity);       //!< fluid vorticity (vort, |vort|)
    // DEFINE_SOA_SET_GET(Real, MT, m_o, Omega);               //!< the norm2 of fluid vorticity
    DEFINE_SOA_SET_GET(vec4r, MT, m_n, Normal);             //!< fluid normal
    DEFINE_SOA_SET_GET(vec4r, MT, m_f, Force);              //!< force
    

    ParticleFluid(Real r, unsigned int size, bool useFoam=false):Base(r,size){
        // Check if size is not zero
        if (size != 0) {
            if (useFoam) {
                m_x.resize(size * 2, false);
            }
            m_txph.resize(size,false);

            m_ph.resize(size, false);
            m_h.resize(size, false);            
            m_rs.resize(size, false);
            m_tsxph.resize(size, false);

            m_s2o.resize(size, false);
            m_o2s.resize(size, false);

            m_sm.resize(size, false);
            // m_sx.resize(size, false);
            m_sv.resize(size, false);
            m_sph.resize(size, false);
            m_sxph.resize(size, false);
            m_dv.resize(size, false);
            
            // m_rho.resize(size, false);
            m_irho.resize(size, false);
            m_cf.resize(size, false);
            m_l.resize(size, false);
            m_vort.resize(size, false);
            m_n.resize(size, false);
            m_f.resize(size, false);
        }
    }

    virtual ~ParticleFluid() {
        LOG_OSTREAM_DEBUG << "release particle fluid" << std::hex << &m_x << std::dec << std::endl;
        m_txph.release();
        
        m_ph.release();
        m_h.release();            
        m_rs.release();
        m_tsxph.release();

        m_s2o.release();
        m_o2s.release();

        m_sm.release();
        // m_sx.release();
        m_sv.release();
        m_sph.release();
        m_sxph.release();
        m_dv.release();
        
        // m_rho.release();
        m_irho.release();
        m_cf.release();
        m_l.release();
        m_vort.release();
        m_n.release();
        m_f.release();
        // m_gradc.release();
        LOG_OSTREAM_DEBUG << "release ParticleFluid finished"<<std::endl;
    }

    void init() {};

    /**
     * @brief Create a stack of particles.
     *
     * @param length The length of the stack in each dimension.
     * @param gap The gap between each particle.
     */
    void stack(vec3r length, Real gap){
        vec3r count=length/gap;
        callStackParticle<MT>(m_x.size(), (int)count.x, (int)count.y, (int)count.z, gap, m_x);
    }

    /**
     * @brief Sets the velocity of the particles.
     * 
     * @param vel The velocity vector.
     */
    void setVelocity(vec3r vel){
        callFillArray<MT>(m_v.size(), vel, m_v);
    }

    /**
     * @brief Apply advection to the given data.
     *
     * @param dt - The time step size.
     */
    void advect(Real dt){
        callAdvect<MT>(m_x.size(), dt, m_x, m_v);
    }

    /**
     * @brief Resizes the PF vectors based on the given size.
     * 
     * @param size The new size for the vectors.
     * @param useFoam Flag indicating whether to use foam.
     */
    void resizePF(unsigned int size, bool useFoam = false){
        // Check if size is not zero
        if (size != 0) {
            if (useFoam)
                m_x.resize(size * 2);
            else
                m_x.resize(size);

            m_tx.resize(size);
            m_txph.resize(size);
            m_v.resize(size);
            m_tv.resize(size);
            m_m.resize(size);

            m_ph.resize(size);
            m_h.resize(size);            
            m_rs.resize(size);
            m_tsxph.resize(size);

            m_s2o.resize(size);
            m_o2s.resize(size);

            m_sm.resize(size);
            // m_sx.resize(size);
            m_sv.resize(size);
            m_sph.resize(size);
            m_sxph.resize(size);
            m_dv.resize(size);

            // m_rho.resize(size);
            m_irho.resize(size);
            m_cf.resize(size);
            m_l.resize(size);
            m_vort.resize(size);
            m_n.resize(size);
            m_f.resize(size);
        }
    };

    /**
     * @brief Clears the velocity elements of a vector.
     * 
     * @tparam MT The type of the vector.
     * @param vec The vector to clear the velocity elements of.
     */
    void clearVelocity(){
        callFillArray<MT>(m_v.size(), make_vec3r(0.,0.,0.), m_v);
    }

    /**
     * @brief Applies a body force to a vector array.
     * 
     * @param bodyForce The body force to apply.
     * @param dt The time step size.
     */
    void applyBodyForce(vec3r bodyForce, Real dt){
        callAddArray<MT>(m_v.size(), bodyForce*dt, m_v);
    }

    /**
     * @brief Applies an attract force to the particles in the system.
     * 
     * @param center The center of attraction.
     * @param scale The scale of the attraction force.
     * @param dt The time step for advection.
     */
    void applyAttractForce(vec3r center, Real scale, Real dt){
        callComputeAttractForce<MT>(m_v.size(), center, scale, m_tx, m_x);
        callAdvect<MT>(m_v.size(), dt, m_v, m_tx);
    }

};
PHYS_NAMESPACE_END