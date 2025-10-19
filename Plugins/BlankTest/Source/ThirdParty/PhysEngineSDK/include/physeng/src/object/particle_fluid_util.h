#pragma once
#include "object/particle_fluid.h"

PHYS_NAMESPACE_BEGIN


/**
 * @brief Checks if a given phase is a liquid or oil.
 * 
 * @param ph The phase to check.
 * @return True if the phase is liquid or oil, false otherwise.
 */
inline __host__ __device__ bool _d_isLiquid(uint ph){
    return ph==(uint)PhaseType::Liquid || ph==(uint)PhaseType::Oil;
}

// inline __host__ __device__ vec3r _d_enforceBoundaryLocal(vec3r pos){
//     vec3r npos=pos;
//     npos.x=(max(min(npos.x,params.worldMax.x), params.worldMin.x)+pos.x)*0.5f;
//     npos.y=(max(min(npos.y,params.worldMax.y), params.worldMin.y)+pos.y)*0.5f;
//     npos.z=(max(min(npos.z,params.worldMax.z), params.worldMin.z)+pos.z)*0.5f;
//     return npos;
// }

/**
 * @brief Calculates the poly6 value for a given distance.
 * 
 * @param d The distance value.
 * @param params The parameters used in the calculation.
 * @return The calculated poly6 value.
 */
inline __host__ __device__ Real _d_poly6(Real d){
    // Calculate the difference between the squared smoothing radius and the squared distance.
    Real tmp = params.h2 - d * d;
    // Calculate the poly6 value using the difference.
    return params.poly6Coef * tmp*tmp*tmp;
}

/**
 * @brief Calculates the gradient of the poly6 function.
 * 
 * @param r The position vector.
 * @param d The distance.
 * @return The gradient vector.
 */
inline __host__ __device__ vec3r _d_poly6Grad(vec3r r, Real d){
    // Calculate the difference between the smoothing length squared and the distance squared
    Real tmp = params.h2 - d*d;
    // Calculate the gradient vector
    return -6 * params.poly6Coef * tmp*tmp*r;
}

/**
 * @brief Calculate the value of the poly6 kernel function for a given distance.
 *
 * @param d - The distance between particles.
 * @param h2 - The square of the smoothing length.
 * @param poly6Coef - The coefficient used in the poly6 kernel function.
 * @return The value of the poly6 kernel function.
 */
inline __host__ __device__ Real _d_poly6(const Real d, const Real h2, const Real poly6Coef){
    Real tmp = params.h2 - d*d;
    return params.poly6Coef * tmp*tmp*tmp;
}

/**
 * @brief Calculates the gradient of the spiky kernel function.
 * 
 * @param r The position vector.
 * @return The gradient vector.
 */
inline __host__ __device__ vec3r _d_spikyGrad(vec3r r){
    // Calculate the distance between the particles
    Real d = length(r);
    // Calculate the difference between the smoothing length and the distance
    Real tmp = params.h - d;
    // Calculate the gradient vector using the spikyGradCoef and the difference
    return (params.spikyGradCoef * tmp * tmp / max(d, 1e-8f)) * r;
}

/**
 * @brief Calculate the gradient of the spiky kernel function.
 *
 * @param r The vector from the particle to its neighbor.
 * @param d The distance between the particle and its neighbor.
 * @return The gradient of the spiky kernel function.
 */
inline __host__ __device__ vec3r _d_spikyGrad(vec3r r, Real d){
    Real tmp = params.h - d;
    return (params.spikyGradCoef / max(d, 1e-8f) * tmp * tmp) * r;
}

/**
 * @brief Calculate the gradient of a spiky kernel function.
 * 
 * @param r The distance vector between two particles.
 * @param d The distance between two particles.
 * @param h The smoothing radius.
 * @param spikyGradCoef The coefficient of the spiky gradient.
 * @return The gradient vector.
 */
inline __host__ __device__ vec3r _d_spikyGrad(const vec3r& r, const Real& d, const Real& h, const Real& spikyGradCoef){
    Real tmp = params.h - d;
    // Calculate the gradient using the spiky gradient formula
    // return (spikyGradCoef * tmp * tmp / max(d, 1e-8f)) * r;
    return (params.spikyGradCoef / (d+1e-8f) * tmp * tmp) * r;
}

//// for cohesion
//// ref to "2013-Versatile surface tension and adhesion for sph fluids"
/**
 * @brief Calculates the cohesion spline value for a given distance.
 *
 * @param d - The distance.
 * @return The cohesion spline value.
 */
inline __host__ __device__ Real _d_cSpline(Real d){
    Real hr3r3=pow((params.h-d)*d,3.0f);
    if(d<params.halfh) return (2*hr3r3-params.cohesionConstCoef)*params.cohesionCoef;
    else return hr3r3*params.cohesionCoef;
}

/**
 * @brief Compute the weight based on the distance.
 * 
 * @param d The distance value.
 * @return The weight value.
 */
inline __host__ __device__ Real _d_rsWeight(Real d){
    if(d<params.h)
        return 1-d/params.h;
    else
        return 0;
}

/**
 * @brief Clamps the value `x` between `Tmin` and `Tmax`, and then normalizes it.
 * 
 * @param x The value to be clamped and normalized.
 * @param Tmin The minimum value.
 * @param Tmax The maximum value.
 * @return The clamped and normalized value.
 */
inline __host__ __device__ Real _d_clampAndNormalize(Real x, Real Tmin, Real Tmax){
    return (min(x, Tmax) - min(x, Tmin))/(Tmax - Tmin);  
}

/**
 * @brief Get two orthogonal vectors in the plane perpendicular to the input vector.
 * @param vec The input vector.
 * @param e1 The first orthogonal vector.
 * @param e2 The second orthogonal vector.
 */
inline __host__ __device__ void _d_getOrthogonalVectors(vec3r vec, vec3r& e1, vec3r& e2) {
    vec3r v = make_vec3r(1, 0, 0);
    
    // If the input vector is parallel to the x-axis, use the y-axis as the reference vector.
    if (fabs(dot(v, vec)) > 0.999)
        v = make_vec3r(0, 1, 0);

    e1 = cross(vec, v);
    e2 = cross(vec, e1);
    e1 = normalize(e1);
    e2 = normalize(e2);
}

PHYS_NAMESPACE_END