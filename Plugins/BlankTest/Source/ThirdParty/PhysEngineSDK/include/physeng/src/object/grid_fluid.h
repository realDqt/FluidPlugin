#pragma once
#include "object/grid_system.h"
#include "common/soa.h"

PHYS_NAMESPACE_BEGIN
template<MemType MT>
class GridFluid : public GridSystem<MT>{
public:
    using Base=GridSystem<MT>;

    //// inherent variables
    USING_BASE_SOA_SET_GET(m_divergence, Divergence);         //!< Base::divergence
    USING_BASE_SOA_SET_GET(m_pressure, Pressure);             //!< Base::pressure
    USING_BASE_SOA_SET_GET(m_tempPressure, TempPressure);     //!< Base::temp pressure
    USING_BASE_SOA_SET_GET(m_density, Density);               //!< Base::density
    USING_BASE_SOA_SET_GET(m_tempDensity, TempDensity);       //!< Base::temp density
    USING_BASE_SOA_SET_GET(m_temperature, Temperature);        //!< Base::temperature
    USING_BASE_SOA_SET_GET(m_tempTemperature, TempTemperature);//!< Base::temp temperature
    USING_BASE_SOA_SET_GET(m_velocity, Velocity);             //!< Base::velocity
    USING_BASE_SOA_SET_GET(m_tempVelocity, TempVelocity);     //!< Base::temp velocity
    USING_BASE_SOA_SET_GET(m_vorticity, Vorticity);           //!< Base::vorticity
    
    GridFluid(uint3 gridSize, Real lengthPerCell):Base(gridSize, lengthPerCell){

    }

    virtual ~GridFluid() {
        LOG_OSTREAM_DEBUG << "release grid fluid" << std::hex << &m_density << std::dec << std::endl;
        
        LOG_OSTREAM_DEBUG << "release grid fluid finished"<<std::endl;
    }

    /**
     * @brief Resizes the GF (Grid Fluid) with the given gridSize and lengthPerCell.
     *
     * @param gridSize the size of the grid in three dimensions (x, y, z)
     * @param lengthPerCell the length per cell in the grid
     *
     * @throws None
     */
    void resizeGF(uint3 gridSize, Real lengthPerCell) {
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
    };

    /**
     * @brief Set the velocity of the object.
     *
     * @param vel the new velocity to set
     *
     * @throws ErrorType description of error
     */
    void setVelocity(vec3r vel){
        callFillArray<MT>(m_velocity.size(), vel, m_velocity);
    }

    /**
     * @brief Clears the velocity array by filling it with zeros.
     *
     * @throws ErrorType description of error
     */
    void clearVelocity(){
        callFillArray<MT>(m_velocity.size(), make_vec3r(0.,0.,0.), m_velocity);
    }

    /**
     * @brief Apply a body force to the velocity array.
     *
     * @param bodyForce the force to be applied to each element of the velocity array
     * @param dt the time step for the force application
     *
     * @throws ErrorType description of error
     */
    void applyBodyForce(vec3r bodyForce, Real dt){
        callAddArray<MT>(m_velocity.size(), bodyForce*dt, m_velocity);
    }

};
PHYS_NAMESPACE_END