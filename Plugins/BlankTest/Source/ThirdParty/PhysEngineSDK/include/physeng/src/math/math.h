#pragma once
// Include necessary headers and define configuration macros for choosing a math library backend.
// Include the general and real headers.

/**
 * @brief This is a math library for performing various mathematical operations.
 *
 * This library provides functions for vector operations, matrix operations,
 * and quaternion operations.
 */
#include "common/real.h"



#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/SparseCore>
#include <Eigen/Geometry>
#include <Eigen/SVD>

// Begin the physics namespace.
PHYS_NAMESPACE_BEGIN
// Conditional type aliases based on whether double precision is used.


/**
 * @brief Type aliases for single precision (float).
 */
using Vector3 = Eigen::Vector3f; /**< @brief 3D vector. */
using Matrix3x3 = Eigen::Matrix3f; /**< @brief 3x3 matrix. */
using Quaternion = Eigen::Quaternionf; /**< @brief Quaternion. */
PHYS_NAMESPACE_END

