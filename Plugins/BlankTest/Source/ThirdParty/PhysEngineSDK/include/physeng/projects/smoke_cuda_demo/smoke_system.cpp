#include "smoke_system.h"
#include "common/timer.h"
#include <algorithm>

using namespace physeng;

SmokeSystem::SmokeSystem(uint3 gridSize, Real cellLength, vec3r worldMin, vec3r worldMax) :
	m_hd(),
	m_ht(),
	m_hv(),
	m_hc(),
	m_c(){

	const float kPi = 3.141592654f;

	m_isInited = false;

	m_solverIterations = 6;
	m_subSteps = 1;
	//// grid info
	m_gridSize = gridSize;
	m_cellLength = cellLength;
	m_numCells = m_gridSize.x * m_gridSize.y * m_gridSize.z;

	//// simulation world
	m_params.worldOrigin = worldMin;
	m_params.worldMin = worldMin;
	m_params.worldMax = worldMax;

	//// grid params
	m_params.gridSize = m_gridSize;
	m_params.numCells = m_numCells;
	m_params.cellLength = cellLength;
	m_params.invCellLength = 1. / cellLength;
	m_params.gridHashMultiplier = make_uint3(1, m_gridSize.x, m_gridSize.x * m_gridSize.y);

	//// physics parameter
	m_params.gravity = make_vec3r(0.0f, -9.8f, 0.0f);
	m_params.kvorticity = 5;
	m_params.kdiffusion = 0.01;
	m_params.bdensity = 0.3;
	m_params.btemperature = 0.2;

	gf.resizeGF(gridSize, cellLength);
	_initialize();
}

SmokeSystem::~SmokeSystem(){
	_finalize();
}

void SmokeSystem::_initialize(){
	assert(!m_isInited);

	//// allocate host storage
	m_hd.resize(m_numCells, false);
	m_hd.fill(m_numCells, 0);
	m_ht.resize(m_numCells, false);
	m_ht.fill(m_numCells, 0);
	m_hv.resize(m_numCells, false);
	m_hv.fill(m_numCells, make_vec3r(0.0f));
	m_hc.resize(m_numCells, false);
	m_hc.fill(m_numCells, make_vec3r(0.9f,0.9f,0.9f));
	m_c.resize(m_numCells, false);

	setGridParameters(&m_params);
	submitToDevice(0, m_numCells);
	m_isInited = true;
}

void
SmokeSystem::_finalize(){
	assert(m_isInited);

	m_hd.release();
	m_ht.release();
	m_hv.release();
	m_hc.release();
	m_c.release();
}

//// step the simulation
void SmokeSystem::update(Real deltaTime) {
	assert(m_isInited);
	Real subDt = deltaTime / m_subSteps;
	int3 source = make_int3(m_gridSize.x / 4, m_gridSize.y / 4, m_gridSize.z / 4);
	int radius2 = 5 * 5;
	Real density = 1;
	PHY_PROFILE("update");
	if (m_numCells > 0) {
		{
			callAddDensity<MemType::GPU>(
				m_numCells,
				source,
				radius2,
				density,
				gf.getDensityRef());
			cudaDeviceSynchronize();
		}
		{
			callAddDensity<MemType::GPU>(
				m_numCells,
				source,
				radius2,
				10,
				gf.getTemperatureRef());
			cudaDeviceSynchronize();
		}
		{
			callDiffuse<MemType::GPU>(
				m_numCells,
				deltaTime,
				gf.getTempDensityRef(),
				gf.getDensityRef());
			cudaDeviceSynchronize();
			gf.getDensityRef().swap(gf.getTempDensityRef());
		}
		{
			callDiffuse<MemType::GPU>(
				m_numCells,
				deltaTime,
				gf.getTempTemperatureRef(),
				gf.getTemperatureRef());
			cudaDeviceSynchronize();
			gf.getTemperatureRef().swap(gf.getTempTemperatureRef());
		}
		//// substeps
		for (uint s = 0; s < m_subSteps; s++) {
			{
				callAdvectProperty<MemType::GPU>(
					m_numCells,
					subDt,
					gf.getTempDensityRef(),
					gf.getDensityRef(),
					gf.getVelocityRef());
				cudaDeviceSynchronize();
				gf.getDensityRef().swap(gf.getTempDensityRef());
			}
			{
				callAdvectVelocity<MemType::GPU>(
					m_numCells,
					subDt,
					gf.getTempVelocityRef(),
					gf.getVelocityRef());
				cudaDeviceSynchronize();
				gf.getVelocityRef().swap(gf.getTempVelocityRef());
			}
			{
				callAddBuoyancy<MemType::GPU>(
					m_numCells,
					subDt,
					gf.getDensityRef(),
					gf.getTemperatureRef(),
					gf.getVelocityRef());
				cudaDeviceSynchronize();
			}
			////// vorticity confinement
			{
				callComputeVorticity<MemType::GPU>(
					m_numCells,
					gf.getVorticityRef(),
					gf.getVelocityRef());
				cudaDeviceSynchronize();
			}
			{
				callConfineVorticity<MemType::GPU>(
					m_numCells,
					subDt,
					gf.getVorticityRef(),
					gf.getVelocityRef());
				cudaDeviceSynchronize();
			}
			{
				callComputeDivergence<MemType::GPU>(
					m_numCells,
					gf.getDivergenceRef(),
					gf.getVelocityRef());
				cudaDeviceSynchronize();
			}
			////// solver iterations
			for (uint i = 0; i < m_solverIterations; i++) {
				{
					callComputePressure<MemType::GPU>(
						m_numCells,
						gf.getDivergenceRef(),
						gf.getTempPressureRef(),
						gf.getPressureRef());
					cudaDeviceSynchronize();
					gf.getPressureRef().swap(gf.getTempPressureRef());
				}
			}
			{
				callProject<MemType::GPU>(
					m_numCells,
					gf.getVelocityRef(),
					gf.getPressureRef());
				cudaDeviceSynchronize();
			}
		}
	}	
}
void SmokeSystem::submitToDevice(uint start, uint count){
	copyArray<Real, MemType::GPU, MemType::CPU>(&gf.getDensityRef().m_data, &m_hd.m_data, start, count);
	copyArray<Real, MemType::GPU, MemType::CPU>(&gf.getTemperatureRef().m_data, &m_ht.m_data, start, count);
	copyArray<vec3r, MemType::GPU, MemType::CPU>(&gf.getVelocityRef().m_data, &m_hv.m_data, start, count);
	copyArray<vec3r, MemType::GPU, MemType::CPU>(&m_c.m_data, &m_hc.m_data, start, count);
}

void SmokeSystem::clear(){
}