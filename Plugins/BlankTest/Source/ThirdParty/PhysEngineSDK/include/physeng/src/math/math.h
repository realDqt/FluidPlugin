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

//// if use backend wrapper
#if PE_USE_BACKEND_WRAPPER
#include <linear_math/vector.h>
#include <linear_math/matrix.h>
#include <linear_math/quaternion.h>
PHYS_NAMESPACE_BEGIN
using Vector3 = linear_math::Vector3;
/**
 * @brief Alias for a 3x3 matrix.
 *
 * Represents a 3x3 matrix.
 *
 * @param m00 The element at row 0, column 0.
 * @param m01 The element at row 0, column 1.
 * @param m02 The element at row 0, column 2.
 * @param m10 The element at row 1, column 0.
 * @param m11 The element at row 1, column 1.
 * @param m12 The element at row 1, column 2.
 * @param m20 The element at row 2, column 0.
 * @param m21 The element at row 2, column 1.
 * @param m22 The element at row 2, column 2.
 */
using Matrix3x3 = linear_math::Matrix3;
/**
 * @brief Alias for a quaternion.
 *
 * Represents a quaternion, which is a mathematical object commonly used in 3D graphics and computer vision.
 */
using Quaternion = linear_math::Quaternion;
PHYS_NAMESPACE_END
//// else use backend wrapper
#else

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/SparseCore>
#include <Eigen/Geometry>
#include <Eigen/SVD>

// Begin the physics namespace.
PHYS_NAMESPACE_BEGIN
// Conditional type aliases based on whether double precision is used.

#if USE_DOUBLE
/**
 * @brief Type aliases for double precision.
 */
using Vector3 = Eigen::Vector3d; /**< @brief 3D vector. */
using Matrix3x3 = Eigen::Matrix3d; /**< @brief 3x3 matrix. */
using Quaternion = Eigen::Quaterniond; /**< @brief Quaternion. */
#else
/**
 * @brief Type aliases for single precision (float).
 */
using Vector3 = Eigen::Vector3f; /**< @brief 3D vector. */
using Matrix3x3 = Eigen::Matrix3f; /**< @brief 3x3 matrix. */
using Quaternion = Eigen::Quaternionf; /**< @brief Quaternion. */
#endif
PHYS_NAMESPACE_END
#endif
