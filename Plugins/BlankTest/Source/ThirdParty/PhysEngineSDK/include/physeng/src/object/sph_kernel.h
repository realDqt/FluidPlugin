#pragma once
#include "common/general.h"
#include "common/array.h"

PHYS_NAMESPACE_BEGIN

class SphKernel{
public:
    SphKernel(Real _h){
        Init(_h);
    }

    // Constructor for SphKernel class
    // Initializes the class with the given smoothing length (_h)
    void Init(Real _h){
        // Set the smoothing length
        h=_h;

        // Compute h^6
        h6=pow(h, 6);

        // Compute h^9
        h9=pow(h, 9);

        // Compute the constant value for the poly6 kernel
        poly6_const=315.0f / (64 * REAL_PI * h9);

        // Precompute constant value for Spiky gradient
        spiky_grad_const=-45.0f / (REAL_PI * h6);

        // Precompute constant value for Cubic spline kernel (k)
        cspline_k_const = 32.0f / (REAL_PI * pow(h, 9));

        // Precompute constant value for Cubic spline kernel (a)
        cspline_a_const = -pow(h, 6) / 64.0f;

        // Precompute constant value for Cubic kernel
        cubic_const = 1 / (REAL_PI * pow(h, 3));
    }

    Real h; // Smoothing length
    Real h6; // h^6
    Real h9; // h^9
    Real poly6_const; // Constant value for Poly6 kernel
    Real spiky_grad_const; // Constant value for Spiky gradient
    Real cspline_k_const; // Constant value for Cubic spline kernel (k)
    Real cspline_a_const; // Constant value for Cubic spline kernel (a)
    Real cubic_const; // Constant value for Cubic kernel

    /**
     * @brief Calculates the Poly6 kernel value for the given deltaPos.
     *
     * @param deltaPos The vector difference between positions.
     * @return The Poly6 kernel value.
     */
    inline PE_CUDA_FUNC Real Poly6(vec3r deltaPos)
    {
        // Check if the squared distance is within the kernel radius
        Real checkVal = h * h - dot(deltaPos,deltaPos);
        // real constant = 315.0f / (64 * REAL_PI * h9);
        return checkVal > 0 ? poly6_const * pow(checkVal, 3) : 0;
    }

    /**
     * @brief Calculates the Spiky gradient for the given deltaPos.
     * 
     * @param deltaPos The vector difference between positions.
     * @return The Spiky gradient vector.
     */
    inline PE_CUDA_FUNC vec3r SpikyGrad(vec3r deltaPos)
    {
        // Calculate the distance between the positions.
        Real r = length(deltaPos);
        // Calculate the difference between the kernel radius and the distance.
        Real checkVal = h - r;
        // real constant = -45.0f / (REAL_PI * h6);

    #ifdef PE_USE_CUDA 
        // If the difference between the kernel radius and the distance is positive,
        // calculate the Spiky gradient using the CUDA implementation.
        // This involves multiplying the constant by the squared difference,
        // dividing by the maximum of the distance and a small epsilon value,
        // and multiplying by the vector difference between positions.
        return checkVal > 0 ? spiky_grad_const * pow(checkVal, 2) * deltaPos / max(r, 1e-8f) : make_vec3r(0, 0, 0);
    #else
        // If the difference between the kernel radius and the distance is positive,
        // calculate the Spiky gradient using the non-CUDA implementation.
        // This involves multiplying the constant by the squared difference,
        // dividing by the maximum of the distance and a small epsilon value,
        // and multiplying by the vector difference between positions.
        return checkVal > 0 ? spiky_grad_const * pow(checkVal, 2) * deltaPos / std::max(r, 1e-8f) : Vector3r(0, 0, 0);
    #endif
    }

    /**
     * @brief Calculates the Cubic spline kernel value for the given deltaPos.
     * 
     * @param deltaPos The vector representing the difference in position.
     * @return The value of the Cubic spline kernel.
     */
    inline PE_CUDA_FUNC Real Cspline(vec3r deltaPos)
    {
        Real r = length(deltaPos);
        // real coefK = 32.0f / (REAL_PI * pow(h, 9));
        // real coefA = -pow(h, 6) / 64.0f;
        if (r > h)
            // Return 0 if the distance is greater than the smoothing length
            return 0; 
        else if (2 * r > h)
            // Calculate the Cubic spline kernel value (k) for 2r > h
            return cspline_k_const * pow((h - r) * r, 3); 
        else
            // Calculate the Cubic spline kernel value (k) for 2r <= h
            return cspline_k_const * (2 * pow((h - r) * r, 3) + cspline_a_const); 
    }

    /**
     * @brief Calculates the Cubic kernel value for the given deltaPos
     *
     * @param deltaPos the difference between two positions
     * @return The Cubic kernel value for the given deltaPos
     */
    inline PE_CUDA_FUNC Real Cubic(vec3r deltaPos)
    {
        // Calculate the distance between the two positions
        Real r = length(deltaPos);
        // Calculate the normalized distance
        Real q = r / h;
        // Check if q is less than 1
        if (q < 1) {
            // Calculate the Cubic kernel value for q < 1
            return cubic_const * (1 - 1.5 * q * q * (1 - 0.5 * q)); 
        }
        // Check if q is less than 2
        else if (q < 2) {
            // Calculate the Cubic kernel value for 1 <= q < 2
            return 0.25 * cubic_const * pow(2 - q, 3); 
        }
        else
            // Return 0 if q >= 2
            return 0; 
    }
};

PHYS_NAMESPACE_END