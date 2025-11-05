
#ifndef CUDA_MATH_H
#define CUDA_MATH_H

#include <math.h>

// #include <nvVector.h>
// #include <nvMatrix.h>
// #include <nvQuaternion.h>

// #include "math/math.h"
#include "common/real.h"
#ifndef USE_DOUBLE
#include "helper_math.h"

#define make_vec2r(...) make_float2(__VA_ARGS__)
#define make_vec3r(...) make_float3(__VA_ARGS__)
#define make_vec4r(...) make_float4(__VA_ARGS__)
#else
static_assert("double unimplemented");
#endif

// typedef vec2<float> vec2f;
// typedef vec3<float> vec3f;
// typedef vec3<int> vec3i;
// typedef vec3<unsigned int> vec3ui;
// typedef vec4<float> vec4f;
// typedef matrix4<float> matrix4f;
// typedef quaternion<float> quaternionf;

#ifndef USE_DOUBLE
typedef float2 vec2r;
typedef float3 vec3r;
typedef float4 vec4r;
#else
    /// cause error now!
    typedef double2 vec2r;
typedef double3 vec3r;
typedef double4 vec4r;
#endif

/**
 * Compute the inverse of a 3x3 matrix.
 * 
 * @param mat - The input matrix.
 * @param det - The determinant of the input matrix.
 * @param inv - The output inverse matrix.
 * @return true if the inverse is successfully computed, false otherwise.
 */
inline bool inverse(const vec3r *mat, Real det, vec3r *inv)
{
  // Compute the determinant of the matrix
  det = mat[0].x * (mat[1].y * mat[2].z - mat[2].y * mat[1].z) -
        mat[0].y * (mat[1].x * mat[2].z - mat[1].z * mat[2].x) +
        mat[0].z * (mat[1].x * mat[2].y - mat[1].y * mat[2].x);

  // Check if the determinant is close to zero
  if (fabs(det) < 1e-8)
    return false;

  // Compute the inverse determinant
  Real invdet = 1 / det;

  // Compute the elements of the inverse matrix
  inv[0].x = (mat[1].y * mat[2].z - mat[2].y * mat[1].z) * invdet;
  inv[0].y = (mat[0].z * mat[2].y - mat[0].y * mat[2].z) * invdet;
  inv[0].z = (mat[0].y * mat[1].z - mat[0].z * mat[1].y) * invdet;
  inv[1].x = (mat[1].z * mat[2].x - mat[1].x * mat[2].z) * invdet;
  inv[1].y = (mat[0].x * mat[2].z - mat[0].z * mat[2].x) * invdet;
  inv[1].z = (mat[1].x * mat[0].z - mat[0].x * mat[1].z) * invdet;
  inv[2].x = (mat[1].x * mat[2].y - mat[2].x * mat[1].y) * invdet;
  inv[2].y = (mat[2].x * mat[0].y - mat[0].x * mat[2].y) * invdet;
  inv[2].z = (mat[0].x * mat[1].y - mat[1].x * mat[0].y) * invdet;

  return true;
}

#include <iostream>
// ref to https://github.com/wyegelwel/snow/blob/b504448296e6c161f25098d12c4b5358220e767a/project/cuda/matrix.h
class mat3r
{
public:
  union
  {
    vec3r rows[3];
    Real data[9];
  };

  /**
   * @brief Initialize a 3x3 matrix with all elements set to zero.
   * 
   * @return The initialized matrix.
   */
  __device__ __host__ __forceinline__ mat3r()
  {
    rows[0] = make_vec3r(0.0f);
    rows[1] = make_vec3r(0.0f);
    rows[2] = make_vec3r(0.0f);
  }

  /**
   * @brief Create a 3x3 matrix with all elements set to a given value.
   * 
   * @param i The value to set for all elements of the matrix.
   * @return The created matrix.
   */
  __device__ __host__ __forceinline__ mat3r(Real i)
  {
    rows[0] = make_vec3r(i, 0.0f, 0.0f);
    rows[1] = make_vec3r(0.0f, i, 0.0f);
    rows[2] = make_vec3r(0.0f, 0.0f, i);
  }

  /**
   * @brief Construct a 3x3 matrix with diagonal elements from a given vector.
   * 
   * @param diag The diagonal vector.
   * @return The constructed matrix.
   */
  __device__ __host__ __forceinline__ mat3r(const vec3r &diag)
  {
    rows[0] = make_vec3r(diag.x, 0.0f, 0.0f);
    rows[1] = make_vec3r(0.0f, diag.y, 0.0f);
    rows[2] = make_vec3r(0.0f, 0.0f, diag.z);
  }

  /**
   * @brief Construct a 3x3 matrix with given rows.
   * 
   * @param r0 The first row vector.
   * @param r1 The second row vector.
   * @param r2 The third row vector.
   * @return The constructed matrix.
   */
  __device__ __host__ __forceinline__ mat3r(const vec3r &r0, const vec3r &r1, const vec3r &r2)
  {
    rows[0] = r0;
    rows[1] = r1;
    rows[2] = r2;
  }

  /**
   * @brief Create a 3x3 matrix with the given elements.
   * 
   * @param a The element at (0,0) position.
   * @param b The element at (0,1) position.
   * @param c The element at (0,2) position.
   * @param d The element at (1,0) position.
   * @param e The element at (1,1) position.
   * @param f The element at (1,2) position.
   * @param g The element at (2,0) position.
   * @param h The element at (2,1) position.
   * @param i The element at (2,2) position.
   * @return The constructed matrix.
   */
  __host__ __device__ __forceinline__
  mat3r(float a, float b, float c, float d, float e, float f, float g, float h, float i)
  {
    data[0] = a;
    data[3] = d;
    data[6] = g;
    data[1] = b;
    data[4] = e;
    data[7] = h;
    data[2] = c;
    data[5] = f;
    data[8] = i;
  }

  __host__ __device__ __forceinline__
      mat3r &
      operator=(const mat3r &rhs)
  {
    data[0] = rhs[0];
    data[3] = rhs[3];
    data[6] = rhs[6];
    data[1] = rhs[1];
    data[4] = rhs[4];
    data[7] = rhs[7];
    data[2] = rhs[2];
    data[5] = rhs[5];
    data[8] = rhs[8];
    return *this;
  }

  __host__ __device__ __forceinline__ Real &operator[](int i) { return data[i]; }
  __host__ __device__ __forceinline__ Real operator[](int i) const { return data[i]; }
  __host__ __device__ __forceinline__ Real &operator()(int i) { return data[i]; }
  __host__ __device__ __forceinline__ Real operator()(int i) const { return data[i]; }
  __host__ __device__ __forceinline__ Real &operator()(int r, int c) { return data[r * 3 + c]; }
  __host__ __device__ __forceinline__ Real operator()(int r, int c) const { return data[r * 3 + c]; }

  /**
   * @brief Multiplies two mat3r matrices.
   * 
   * @param rhs The right-hand side matrix.
   * @return A reference to the current matrix after multiplication.
   */
  __host__ __device__ __forceinline__
      mat3r &
      operator*=(const mat3r &rhs)
  {
    mat3r tmp;
    // tmp[0] = data[0]*rhs[0] + data[3]*rhs[1] + data[6]*rhs[2];
    // tmp[1] = data[1]*rhs[0] + data[4]*rhs[1] + data[7]*rhs[2];
    // tmp[2] = data[2]*rhs[0] + data[5]*rhs[1] + data[8]*rhs[2];
    // tmp[3] = data[0]*rhs[3] + data[3]*rhs[4] + data[6]*rhs[5];
    // tmp[4] = data[1]*rhs[3] + data[4]*rhs[4] + data[7]*rhs[5];
    // tmp[5] = data[2]*rhs[3] + data[5]*rhs[4] + data[8]*rhs[5];
    // tmp[6] = data[0]*rhs[6] + data[3]*rhs[7] + data[6]*rhs[8];
    // tmp[7] = data[1]*rhs[6] + data[4]*rhs[7] + data[7]*rhs[8];
    // tmp[8] = data[2]*rhs[6] + data[5]*rhs[7] + data[8]*rhs[8];

    tmp[0] = rhs[0] * data[0] + rhs[3] * data[1] + rhs[6] * data[2];
    tmp[1] = rhs[1] * data[0] + rhs[4] * data[1] + rhs[7] * data[2];
    tmp[2] = rhs[2] * data[0] + rhs[5] * data[1] + rhs[8] * data[2];
    tmp[3] = rhs[0] * data[3] + rhs[3] * data[4] + rhs[6] * data[5];
    tmp[4] = rhs[1] * data[3] + rhs[4] * data[4] + rhs[7] * data[5];
    tmp[5] = rhs[2] * data[3] + rhs[5] * data[4] + rhs[8] * data[5];
    tmp[6] = rhs[0] * data[6] + rhs[3] * data[7] + rhs[6] * data[8];
    tmp[7] = rhs[1] * data[6] + rhs[4] * data[7] + rhs[7] * data[8];
    tmp[8] = rhs[2] * data[6] + rhs[5] * data[7] + rhs[8] * data[8];
    return (*this = tmp);
  }

  /**
   * @brief Multiplies two mat3r matrices.
   * 
   * @param rhs The right-hand side matrix.
   * @return The result of the matrix multiplication.
   */
  __host__ __device__ __forceinline__
      mat3r
      operator*(const mat3r &rhs) const
  {
    mat3r result;
    // result[0] = data[0]*rhs[0] + data[3]*rhs[1] + data[6]*rhs[2];
    // result[1] = data[1]*rhs[0] + data[4]*rhs[1] + data[7]*rhs[2];
    // result[2] = data[2]*rhs[0] + data[5]*rhs[1] + data[8]*rhs[2];
    // result[3] = data[0]*rhs[3] + data[3]*rhs[4] + data[6]*rhs[5];
    // result[4] = data[1]*rhs[3] + data[4]*rhs[4] + data[7]*rhs[5];
    // result[5] = data[2]*rhs[3] + data[5]*rhs[4] + data[8]*rhs[5];
    // result[6] = data[0]*rhs[6] + data[3]*rhs[7] + data[6]*rhs[8];
    // result[7] = data[1]*rhs[6] + data[4]*rhs[7] + data[7]*rhs[8];
    // result[8] = data[2]*rhs[6] + data[5]*rhs[7] + data[8]*rhs[8];
    result[0] = rhs[0] * data[0] + rhs[3] * data[1] + rhs[6] * data[2];
    result[1] = rhs[1] * data[0] + rhs[4] * data[1] + rhs[7] * data[2];
    result[2] = rhs[2] * data[0] + rhs[5] * data[1] + rhs[8] * data[2];
    result[3] = rhs[0] * data[3] + rhs[3] * data[4] + rhs[6] * data[5];
    result[4] = rhs[1] * data[3] + rhs[4] * data[4] + rhs[7] * data[5];
    result[5] = rhs[2] * data[3] + rhs[5] * data[4] + rhs[8] * data[5];
    result[6] = rhs[0] * data[6] + rhs[3] * data[7] + rhs[6] * data[8];
    result[7] = rhs[1] * data[6] + rhs[4] * data[7] + rhs[7] * data[8];
    result[8] = rhs[2] * data[6] + rhs[5] * data[7] + rhs[8] * data[8];
    return result;
  }

  /**
   * @brief This function multiplies a 3D vector by a matrix and returns the result.
   * 
   * @param rhs The vector to multiply with the matrix.
   * @return The resulting vector after multiplication.
   */
  __host__ __device__ __forceinline__
      vec3r
      operator*(const vec3r &rhs) const
  {
    vec3r result;
    // result.x = data[0]*rhs.x + data[3]*rhs.y + data[6]*rhs.z;
    // result.y = data[1]*rhs.x + data[4]*rhs.y + data[7]*rhs.z;
    // result.z = data[2]*rhs.x + data[5]*rhs.y + data[8]*rhs.z;
    result.x = data[0] * rhs.x + data[1] * rhs.y + data[2] * rhs.z;
    result.y = data[3] * rhs.x + data[4] * rhs.y + data[5] * rhs.z;
    result.z = data[6] * rhs.x + data[7] * rhs.y + data[8] * rhs.z;
    return result;
  }

  /**
    * @brief This function adds a matrix to the current matrix in place.
    * 
    * @param rhs The matrix to add.
    * @return A reference to the current matrix after addition.
    */
  __host__ __device__ __forceinline__
      mat3r &
      operator+=(const mat3r &rhs)
  {
    data[0] += rhs[0];
    data[3] += rhs[3];
    data[6] += rhs[6];
    data[1] += rhs[1];
    data[4] += rhs[4];
    data[7] += rhs[7];
    data[2] += rhs[2];
    data[5] += rhs[5];
    data[8] += rhs[8];
    return *this;
  }

  /**
   * @brief Overloads the addition operator for mat3r objects.
   * 
   * @param rhs The right-hand side matrix to be added.
   * @return The sum of the two matrices.
   */
  __host__ __device__ __forceinline__
      mat3r
      operator+(const mat3r &rhs) const
  {
    mat3r tmp = *this;
    tmp[0] += rhs[0];
    tmp[3] += rhs[3];
    tmp[6] += rhs[6];
    tmp[1] += rhs[1];
    tmp[4] += rhs[4];
    tmp[7] += rhs[7];
    tmp[2] += rhs[2];
    tmp[5] += rhs[5];
    tmp[8] += rhs[8];
    return tmp;
  }

  /**
    * @brief Subtract the elements of the given matrix from the corresponding elements of this matrix
    * 
    * @param rhs The matrix to subtract from this matrix
    * @return A reference to the current matrix after subtraction
    */
  __host__ __device__ __forceinline__
      mat3r &
      operator-=(const mat3r &rhs)
  {
    data[0] -= rhs[0];
    data[3] -= rhs[3];
    data[6] -= rhs[6];
    data[1] -= rhs[1];
    data[4] -= rhs[4];
    data[7] -= rhs[7];
    data[2] -= rhs[2];
    data[5] -= rhs[5];
    data[8] -= rhs[8];
    return *this;
  }

  /**
    * @brief Subtract the elements of the given matrix from the corresponding elements of this matrix
    * 
    * @param rhs The matrix to subtract from this matrix
    * @return The resulting matrix after subtraction
    */
  __host__ __device__ __forceinline__
      mat3r
      operator-(const mat3r &rhs) const
  {
    mat3r tmp = *this;
    tmp[0] -= rhs[0];
    tmp[3] -= rhs[3];
    tmp[6] -= rhs[6];
    tmp[1] -= rhs[1];
    tmp[4] -= rhs[4];
    tmp[7] -= rhs[7];
    tmp[2] -= rhs[2];
    tmp[5] -= rhs[5];
    tmp[8] -= rhs[8];
    return tmp;
  }

  /**
   * @brief Multiplies each element of the matrix by a scalar value.
   * 
   * @param f The scalar value to multiply the matrix by.
   * @return A reference to the current matrix after multiplication.
   */
  __host__ __device__ __forceinline__
      mat3r &
      operator*=(float f)
  {
    data[0] *= f;
    data[3] *= f;
    data[6] *= f;
    data[1] *= f;
    data[4] *= f;
    data[7] *= f;
    data[2] *= f;
    data[5] *= f;
    data[8] *= f;
    return *this;
  }

  /**
   * @brief Multiplies each element of the matrix by a scalar value.
   * 
   * @param f The scalar value to multiply the matrix by.
   * @return The resulting matrix after multiplication.
   */
  __host__ __device__ __forceinline__
      mat3r
      operator*(float f) const
  {
    mat3r tmp = *this;
    tmp[0] *= f;
    tmp[3] *= f;
    tmp[6] *= f;
    tmp[1] *= f;
    tmp[4] *= f;
    tmp[7] *= f;
    tmp[2] *= f;
    tmp[5] *= f;
    tmp[8] *= f;
    return tmp;
  }

  /**
   * @brief Divide each element of the matrix by a scalar value.
   * 
   * @param f The scalar value to divide by.
   * @return A reference to the current matrix after dividing.
   */
  __host__ __device__ __forceinline__
      mat3r &
      operator/=(float f)
  {
    float fi = 1.f / f;
    data[0] *= fi;
    data[3] *= fi;
    data[6] *= fi;
    data[1] *= fi;
    data[4] *= fi;
    data[7] *= fi;
    data[2] *= fi;
    data[5] *= fi;
    data[8] *= fi;
    return *this;
  }

  /**
   * @brief Divide each element of the matrix by a scalar value.
   * 
   * @param f The scalar value to divide by.
   * @return The resulting matrix after dividing.
   */
  __host__ __device__ __forceinline__
      mat3r
      operator/(float f) const
  {
    mat3r tmp = *this;
    float fi = 1.f / f;
    tmp[0] *= fi;
    tmp[3] *= fi;
    tmp[6] *= fi;
    tmp[1] *= fi;
    tmp[4] *= fi;
    tmp[7] *= fi;
    tmp[2] *= fi;
    tmp[5] *= fi;
    tmp[8] *= fi;
    return tmp;
  }

  /**
    * @brief Transposes a 3x3 matrix.
    *
    * @return The transposed matrix.
    */
  __host__ __device__ __forceinline__
      mat3r
      transpose() const
  {
    return mat3r(data[0], data[3], data[6],
                 data[1], data[4], data[7],
                 data[2], data[5], data[8]);
  }

  /**
   * @brief Calculate the trace of a 3x3 matrix.
   * 
   * @return The trace of the matrix.
   */
  __host__ __device__ __forceinline__
      Real
      trace() const
  {
    return data[0] + data[4] + data[8];
  }

  /**
    * @brief Calculate the sum of the squares of the elements in the data array.
    *
    * @return The sum of the squares of the elements.
    */
  __host__ __device__ __forceinline__
      Real
      sum2() const
  {
    return data[0] * data[0] + data[3] * data[3] + data[6] * data[6] +
           data[1] * data[1] + data[4] * data[4] + data[7] * data[7] +
           data[2] * data[2] + data[5] * data[5] + data[8] * data[8];
  }

  /**
   * @brief Transposes a 3x3 matrix
   * 
   * @param m The input matrix
   * @return The transposed matrix
   */
  __host__ __device__ __forceinline__ static mat3r transpose(const mat3r &m)
  {
    return mat3r(m[0], m[3], m[6],
                 m[1], m[4], m[7],
                 m[2], m[5], m[8]);
  }

  // __host__ __device__ __forceinline__
  // static mat3r inverse( const mat3r &M )
  // {
  //     Real det = (M[0]*(M[4]*M[8]-M[7]*M[5]) -
  //                 M[3]*(M[1]*M[8]-M[7]*M[2]) +
  //                 M[6]*(M[1]*M[5]-M[4]*M[2]));
  //     mat3r A;
  //     if(fabs(det)<1e-8) return A;
  //     Real invDet = 1.0 / det;
  //     A[0] = invDet * (M[4]*M[8]-M[5]*M[7]);
  //     A[1] = invDet * (M[2]*M[7]-M[1]*M[8]);
  //     A[2] = invDet * (M[1]*M[5]-M[2]*M[4]);
  //     A[3] = invDet * (M[5]*M[6]-M[3]*M[8]);
  //     A[4] = invDet * (M[0]*M[8]-M[2]*M[6]);
  //     A[5] = invDet * (M[2]*M[3]-M[0]*M[5]);
  //     A[6] = invDet * (M[3]*M[7]-M[4]*M[6]);
  //     A[7] = invDet * (M[1]*M[6]-M[0]*M[7]);
  //     A[8] = invDet * (M[0]*M[4]-M[1]*M[3]);
  //     return A;
  // }

  /**
   * @brief Calculate the inverse of a 3x3 matrix
   * 
   * @param mat The input matrix
   * @return The inverse matrix of mat
   */
  __host__ __device__ __forceinline__ static mat3r inverse(const mat3r &mat)
  {
    // Calculate the determinant of the matrix
    Real det = mat(0, 0) * (mat(1, 1) * mat(2, 2) - mat(2, 1) * mat(1, 2)) -
               mat(0, 1) * (mat(1, 0) * mat(2, 2) - mat(1, 2) * mat(2, 0)) +
               mat(0, 2) * (mat(1, 0) * mat(2, 1) - mat(1, 1) * mat(2, 0));

    mat3r inv;

    // If the determinant is close to zero, return an empty matrix
    if (fabs(det) < 1e-8)
      return inv;

    // Calculate the inverse determinant
    Real invdet = 1 / det;

    // Calculate the elements of the inverse matrix
    inv(0, 0) = (mat(1, 1) * mat(2, 2) - mat(2, 1) * mat(1, 2)) * invdet;
    inv(0, 1) = (mat(0, 2) * mat(2, 1) - mat(0, 1) * mat(2, 2)) * invdet;
    inv(0, 2) = (mat(0, 1) * mat(1, 2) - mat(0, 2) * mat(1, 1)) * invdet;
    inv(1, 0) = (mat(1, 2) * mat(2, 0) - mat(1, 0) * mat(2, 2)) * invdet;
    inv(1, 1) = (mat(0, 0) * mat(2, 2) - mat(0, 2) * mat(2, 0)) * invdet;
    inv(1, 2) = (mat(1, 0) * mat(0, 2) - mat(0, 0) * mat(1, 2)) * invdet;
    inv(2, 0) = (mat(1, 0) * mat(2, 1) - mat(2, 0) * mat(1, 1)) * invdet;
    inv(2, 1) = (mat(2, 0) * mat(0, 1) - mat(0, 0) * mat(2, 1)) * invdet;
    inv(2, 2) = (mat(0, 0) * mat(1, 1) - mat(1, 0) * mat(0, 1)) * invdet;

    return inv;
  }

  /**
   * @brief Calculates the inverse of a 3x3 matrix.
   * 
   * @return The inverse of the matrix.
   */
  __host__ __device__ __forceinline__
      mat3r
      inverse() const
  {
    // Calculate the determinant of the matrix
    Real det = data[0] * (data[4] * data[8] - data[7] * data[5]) -
               data[1] * (data[3] * data[8] - data[5] * data[6]) +
               data[2] * (data[3] * data[7] - data[4] * data[6]);

    mat3r inv;

    // Check if the determinant is close to zero
    if (fabs(det) < 1e-8)
      return inv;

    // Calculate the inverse determinant
    Real invdet = 1 / det;

    // Calculate the elements of the inverse matrix
    inv(0, 0) = (data[4] * data[8] - data[7] * data[5]) * invdet;
    inv(0, 1) = (data[2] * data[7] - data[1] * data[8]) * invdet;
    inv(0, 2) = (data[1] * data[5] - data[2] * data[4]) * invdet;
    inv(1, 0) = (data[5] * data[6] - data[3] * data[8]) * invdet;
    inv(1, 1) = (data[0] * data[8] - data[2] * data[6]) * invdet;
    inv(1, 2) = (data[3] * data[2] - data[0] * data[5]) * invdet;
    inv(2, 0) = (data[3] * data[7] - data[6] * data[4]) * invdet;
    inv(2, 1) = (data[6] * data[1] - data[0] * data[7]) * invdet;
    inv(2, 2) = (data[0] * data[4] - data[3] * data[1]) * invdet;

    return inv;
  }

  friend std::ostream &operator<<(std::ostream &os, const mat3r &m);
};

class quaternionf
{
public:
  union
  {
    vec4r _array;
  };

  /**
   * @brief Create a quaternion with all components set to zero.
   * 
   * @return quaternionf The newly created quaternion.
   */
  __host__ __device__ __forceinline__ quaternionf()
  {
    _array = make_vec4r(0.0, 0.0, 0.0, 0.0);
  }

  /**
   * @brief Creates a quaternion from an axis and an angle.
   *
   * @param axis The axis of rotation.
   * @param radians The angle in radians.
   */
  __host__ __device__ __forceinline__ quaternionf(const vec3r &axis, Real radians)
  {
    set_value(axis, radians);
  }

  /**
   * @brief Creates a quaternion from a point.
   *
   * @param point The point to create the quaternion from.
   */
  __host__ __device__ __forceinline__ quaternionf(const vec3r &point)
  {
    _array.x = point.x;
    _array.y = point.y;
    _array.z = point.z;
    _array.w = 0;
  }

  /**
   * @brief Create a quaternion from an array of values.
   *
   * @param v The array of values representing the quaternion.
   * @return quaternionf The newly created quaternion.
   */
  __host__ __device__ __forceinline__ quaternionf(const Real v[4])
  {
    set_value(v);
  }

  /**
   * @brief Copy constructor for the quaternionf class.
   * 
   * @param v The quaternionf object to be copied.
   */
  __host__ __device__ __forceinline__ quaternionf(const quaternionf &v)
  {
    set_value(v);
  }

  /**
   * @brief Constructor for the quaternionf class.
   * 
   * @param q0 The first component of the quaternion.
   * @param q1 The second component of the quaternion.
   * @param q2 The third component of the quaternion.
   * @param q3 The fourth component of the quaternion.
   */
  __host__ __device__ __forceinline__ quaternionf(Real q0, Real q1, Real q2, Real q3)
  {
    set_value(q0, q1, q2, q3);
  }

  // __host__ __device__ __forceinline__ quaternionf(const mat4r &m)
  // {
  //   set_value(m);
  // }

  /**
   * @brief Construct a quaternion from two 3D vectors representing rotation from one vector to another.
   * 
   * @param rotateFrom The vector to rotate from.
   * @param rotateTo The vector to rotate to.
   */
  __host__ __device__ __forceinline__ quaternionf(const vec3r &rotateFrom, const vec3r &rotateTo)
  {
    set_value(rotateFrom, rotateTo);
  }

  // __host__ __device__ __forceinline__ quaternionf(const vec3r &from_look, const vec3r &from_up,
  //                                                 const vec3r &to_look, const vec3r &to_up)
  // {
  //   set_value(from_look, from_up, to_look, to_up);
  // }

  /**
   * @brief Get the value of the quaternion as a vec4r.
   * 
   * @return The value of the quaternion as a vec4r.
   */
  __host__ __device__ __forceinline__ const vec4r get_value() const
  {
    return _array;
  }

  /**
   * @brief Get the values of the quaternion components.
   *
   * @param[out] q0 - The value of the first component of the quaternion.
   * @param[out] q1 - The value of the second component of the quaternion.
   * @param[out] q2 - The value of the third component of the quaternion.
   * @param[out] q3 - The value of the fourth component of the quaternion.
   */
  __host__ __device__ __forceinline__ void get_value(Real &q0, Real &q1, Real &q2, Real &q3) const
  {
    q0 = _array.x;
    q1 = _array.y;
    q2 = _array.z;
    q3 = _array.w;
  }

  /**
   * @brief Set the values of the quaternion components.
   *
   * @param[in] q0 - The value of the first component of the quaternion.
   * @param[in] q1 - The value of the second component of the quaternion.
   * @param[in] q2 - The value of the third component of the quaternion.
   * @param[in] q3 - The value of the fourth component of the quaternion.
   */
  __host__ __device__ __forceinline__ quaternionf &set_value(Real q0, Real q1, Real q2, Real q3)
  {
    _array.x = q0;
    _array.y = q1;
    _array.z = q2;
    _array.w = q3;
    return *this;
  }

  /**
   * @brief Get the axis and radians from a quaternion.
   *
   * @param axis The output axis vector.
   * @param radians The output radians value.
   */
  __host__ __device__ __forceinline__ void get_value(vec3r &axis, Real &radians) const
  {
    radians = Real(acos(_array.w) * Real(2.0));

    if (radians == Real(0.0))
    {
      axis = make_vec3r(0.0, 0.0, 1.0);
    }
    else
    {
      axis.x = _array.x;
      axis.y = _array.y;
      axis.z = _array.z;
      axis = normalize(axis);
    }
  }

  __host__ __device__ __forceinline__ Real &x()
  {
    return _array.x;
  }

  __host__ __device__ __forceinline__ Real &y()
  {
    return _array.y;
  }

  __host__ __device__ __forceinline__ Real &z()
  {
    return _array.z;
  }

  __host__ __device__ __forceinline__ Real &w()
  {
    return _array.w;
  }

  // __host__ __device__ __forceinline__ void get_value(matrix4<Real> &m) const
  // {
  //   Real s, xs, ys, zs, wx, wy, wz, xx, xy, xz, yy, yz, zz;

  //   Real norm = _array[0] * _array[0] + _array[1] * _array[1] + _array[2] * _array[2] + _array[3] * _array[3];

  //   s = (norm == Real(0.0)) ? Real(0.0) : (Real(2.0) / norm);

  //   xs = _array[0] * s;
  //   ys = _array[1] * s;
  //   zs = _array[2] * s;

  //   wx = _array[3] * xs;
  //   wy = _array[3] * ys;
  //   wz = _array[3] * zs;

  //   xx = _array[0] * xs;
  //   xy = _array[0] * ys;
  //   xz = _array[0] * zs;

  //   yy = _array[1] * ys;
  //   yz = _array[1] * zs;
  //   zz = _array[2] * zs;

  //   m(0, 0) = Real(Real(1.0) - (yy + zz));
  //   m(1, 0) = Real(xy + wz);
  //   m(2, 0) = Real(xz - wy);

  //   m(0, 1) = Real(xy - wz);
  //   m(1, 1) = Real(Real(1.0) - (xx + zz));
  //   m(2, 1) = Real(yz + wx);

  //   m(0, 2) = Real(xz + wy);
  //   m(1, 2) = Real(yz - wx);
  //   m(2, 2) = Real(Real(1.0) - (xx + yy));

  //   m(3, 0) = m(3, 1) = m(3, 2) = m(0, 3) = m(1, 3) = m(2, 3) = Real(0.0);
  //   m(3, 3) = Real(1.0);
  // }

  /**
    * @brief Set the value of the quaternion from an array.
    *
    * @param qp The input array containing the values for the quaternion.
    * @return A reference to the modified quaternion.
    */
  __host__ __device__ __forceinline__ quaternionf &set_value(const Real *qp)
  {

    _array.x = qp[0];
    _array.y = qp[1];
    _array.z = qp[2];
    _array.w = qp[3];

    return *this;
  }

  /**
    * @brief Set the value of the quaternion from another quaternion.
    *
    * @param v The input quaternion to copy the value from.
    * @return A reference to the modified quaternion.
    */
  __host__ __device__ __forceinline__ quaternionf &set_value(const quaternionf &v)
  {
    _array = v._array;
    return *this;
  }

  // __host__ __device__ __forceinline__ quaternionf &set_value(const matrix4<Real> &m)
  // {
  //   Real tr, s;
  //   int i, j, k;
  //   const int nxt[3] = {1, 2, 0};

  //   tr = m(0, 0) + m(1, 1) + m(2, 2);

  //   if (tr > Real(0))
  //   {
  //     s = Real(sqrt(tr + m(3, 3)));
  //     _array[3] = Real(s * 0.5);
  //     s = Real(0.5) / s;

  //     _array[0] = Real((m(1, 2) - m(2, 1)) * s);
  //     _array[1] = Real((m(2, 0) - m(0, 2)) * s);
  //     _array[2] = Real((m(0, 1) - m(1, 0)) * s);
  //   }
  //   else
  //   {
  //     i = 0;

  //     if (m(1, 1) > m(0, 0))
  //     {
  //       i = 1;
  //     }

  //     if (m(2, 2) > m(i, i))
  //     {
  //       i = 2;
  //     }

  //     j = nxt[i];
  //     k = nxt[j];

  //     s = Real(sqrt((m(i, j) - (m(j, j) + m(k, k))) + Real(1.0)));

  //     _array[i] = Real(s * 0.5);
  //     s = Real(0.5 / s);

  //     _array[3] = Real((m(j, k) - m(k, j)) * s);
  //     _array[j] = Real((m(i, j) + m(j, i)) * s);
  //     _array[k] = Real((m(i, k) + m(k, i)) * s);
  //   }

  //   return *this;
  // }


  __host__ __device__ __forceinline__ quaternionf &set_value(const vec3r &axis, Real theta)
  {
    Real sqnorm = dot(axis, axis);

    if (sqnorm == Real(0.0))
    {
      // axis too small.
      _array.x = _array.y = _array.z = Real(0.0);
      _array.w = Real(1.0);
    }
    else
    {
      theta *= Real(0.5);
      Real sin_theta = Real(sin(theta));

      if (sqnorm != Real(1))
      {
        sin_theta /= Real(sqrt(sqnorm));
      }

      _array.x = sin_theta * axis.x;
      _array.y = sin_theta * axis.y;
      _array.z = sin_theta * axis.z;
      _array.w = Real(cos(theta));
    }

    return *this;
  }

  __host__ __device__ __forceinline__ quaternionf &set_value(const vec3r &rotateFrom, const vec3r &rotateTo)
  {
    vec3r p1, p2;
    Real alpha;

    p1 = normalize(rotateFrom);
    p2 = normalize(rotateTo);

    alpha = dot(p1, p2);
    if (alpha == Real(1.0))
    {
      *this = quaternionf();
      return *this;
    }

    // ensures that the anti-parallel case leads to a positive dot
    if (alpha == Real(-1.0))
    {
      vec3r v;

      if (p1.x != p1.y || p1.x != p1.z)
      {
        v = make_vec3r(p1.y, p1.z, p1.x);
      }
      else
      {
        v = make_vec3r(-p1.x, p1.y, p1.z);
      }

      v -= p1 * dot(p1, v);
      v = normalize(v);

      set_value(v, Real(3.1415926));
      return *this;
    }

    p1 = normalize(cross(p1, p2));

    set_value(p1, Real(acos(alpha)));

    return *this;
  }

  // __host__ __device__ __forceinline__ quaternionf &set_value(const vec3r &from_look, const vec3r &from_up,
  //                       const vec3r &to_look, const vec3r &to_up)
  // {
  //   quaternionf r_look = quaternionf(from_look, to_look);

  //   vec3r rotated_from_up(from_up);
  //   r_look.mult_vec(rotated_from_up);

  //   quaternionf r_twist = quaternionf(rotated_from_up, to_up);

  //   *this = r_twist;
  //   *this *= r_look;
  //   return *this;
  // }

  __host__ __device__ __forceinline__ quaternionf &operator*=(quaternionf &qr)
  {
    quaternionf ql(*this);
    _array.w = ql.w() * qr.w() - ql.x() * qr.x() - ql.y() * qr.y() - ql.z() * qr.z();
    _array.x = ql.w() * qr.x() + ql.x() * qr.w() + ql.y() * qr.z() - ql.z() * qr.y();
    _array.y = ql.w() * qr.y() + ql.y() * qr.w() + ql.z() * qr.x() - ql.x() * qr.z();
    _array.z = ql.w() * qr.z() + ql.z() * qr.w() + ql.x() * qr.y() - ql.y() * qr.x();
    return *this;
  }

  /**
   * @brief Normalize a quaternion
   * 
   * @param q The quaternion to normalize
   * @return The normalized quaternion
   */
  __host__ __device__ __forceinline__ friend quaternionf normalize(quaternionf &q)
  {
    quaternionf r(q);

    // Calculate the norm of the quaternion
    Real rnorm = Real(1.0) / Real(sqrt(q.w() * q.w() + q.x() * q.x() + q.y() * q.y() + q.z() * q.z()));

    // Normalize the quaternion components
    r._array.x *= rnorm;
    r._array.y *= rnorm;
    r._array.z *= rnorm;
    r._array.w *= rnorm;
    
    return r;
  }

  /**
   * @brief Calculate the squared norm of a quaternion.
   * 
   * @param q The quaternion for which to calculate the squared norm.
   * @return The squared norm of the quaternion.
   */
  __host__ __device__ __forceinline__ friend Real squaredNorm(quaternionf &q)
  {
    return Real(q.w() * q.w() + q.x() * q.x() + q.y() * q.y() + q.z() * q.z());
  }

  /**
   * @brief Calculate the conjugate of a quaternion.
   *
   * @param q The input quaternion.
   * @return The conjugate of the input quaternion.
   */
  __host__ __device__ __forceinline__ friend quaternionf conjugate(const quaternionf &q)
  {
    quaternionf r(q);
    r._array.x *= Real(-1.0);
    r._array.y *= Real(-1.0);
    r._array.z *= Real(-1.0);
    return r;
  }

  /**
   * @brief Calculate the inverse of a quaternion
   * 
   * @param q The input quaternion
   * @return The inverse of the input quaternion
   */
  __host__ __device__ __forceinline__ friend quaternionf inverse(const quaternionf &q)
  {
    return conjugate(q);
  }

  //
  // quaternionf multiplication with cartesian vector
  // v' = q*v*q(star)
  //

  /**
   * @brief Multiply a vector by a quaternion.
   * 
   * This function multiplies a source vector by a quaternion and stores the result in the destination vector.
   * 
   * @param src The source vector.
   * @param dst The destination vector.
   */
  __host__ __device__ __forceinline__ void mult_vec(vec3r &src, vec3r &dst)
  {
    // Calculate coefficients
    Real v_coef = _array.w * _array.w - _array.x * _array.x - _array.y * _array.y - _array.z * _array.z;
    Real u_coef = Real(2.0) * (src.x * _array.x + src.y * _array.y + src.z * _array.z);
    Real c_coef = Real(2.0) * _array.w;

    dst.x = v_coef * src.x + u_coef * x() + c_coef * (y() * src.z - z() * src.y);
    dst.y = v_coef * src.y + u_coef * y() + c_coef * (z() * src.x - x() * src.z);
    dst.z = v_coef * src.z + u_coef * z() + c_coef * (x() * src.y - y() * src.x);
  }

  // __host__ __device__ __forceinline__ void mult_vec(vec3r &src_and_dst) const
  // {
  //   mult_vec(vec3r(src_and_dst), src_and_dst);
  // }

  /**
   * @brief Scale the angle of the axis vector by a given factor.
   * 
   * @param scaleFactor The factor by which to scale the angle.
   */
  __host__ __device__ __forceinline__ void scale_angle(Real scaleFactor)
  {
    vec3r axis;
    Real radians;

    // Get the current axis vector and angle in radians
    get_value(axis, radians);

    // Scale the angle by the given factor
    radians *= scaleFactor;

    // Set the new angle value for the axis vector
    set_value(axis, radians);
  }

  /**
   * Perform spherical linear interpolation (slerp) between two quaternions.
   *
   * @param p The first quaternion.
   * @param q The second quaternion.
   * @param alpha The interpolation parameter.
   * @return The interpolated quaternion.
   */
  __host__ __device__ __forceinline__ friend quaternionf slerp(quaternionf &p, quaternionf &q, Real alpha)
  {
    quaternionf r;

    // the dot product of the two quaternions
    Real cos_omega = p.x() * q.x() + p.y() * q.y() + p.z() * q.z() + p.w() * q.w();
    
    // if B is on opposite hemisphere from A, use -B instead
    int bflip;
    if ((bflip = (cos_omega < Real(0))))
    {
      cos_omega = -cos_omega;
    }

    // complementary interpolation parameter
    Real beta = Real(1) - alpha;

    // If the dot product is greater than or equal to 1, return p
    if (cos_omega >= Real(1))
    {
      return p;
    }

    // the angle between the two quaternions and its reciprocal
    Real omega = Real(acos(cos_omega));
    Real one_over_sin_omega = Real(1.0) / Real(sin(omega));

    // the interpolated values
    beta = Real(sin(omega * beta) * one_over_sin_omega);
    alpha = Real(sin(omega * alpha) * one_over_sin_omega);

    // Flip the sign of alpha if B is on the opposite hemisphere from A
    if (bflip)
    {
      alpha = -alpha;
    }

    // Perform the interpolation
    r._array.x = beta * p._array.x + alpha * q._array.x;
    r._array.y = beta * p._array.y + alpha * q._array.y;
    r._array.z = beta * p._array.z + alpha * q._array.z;
    r._array.w = beta * p._array.w + alpha * q._array.w;

    return r;
  }

  __host__ __device__ __forceinline__ vec4r &coeffs()
  {
    return _array;
  }

  // __host__ __device__ __forceinline__ Real &operator[](int i)
  // {
  //   return _array[i];
  // }

  // __host__ __device__ __forceinline__ const Real &operator[](int i) const
  // {
  //   return _array[i];
  // }

  /**
   * @brief Checks if two quaternions are equal.
   *
   * @param lhs The left-hand side quaternion.
   * @param rhs The right-hand side quaternion.
   * @return True if the quaternions are equal, false otherwise.
   */
  __host__ __device__ __forceinline__ friend bool operator==(const quaternionf &lhs, const quaternionf &rhs)
  {
    bool r = true;

    // Check if each component of the quaternions are equal
    r &= lhs._array.x == rhs._array.x;
    r &= lhs._array.y == rhs._array.y;
    r &= lhs._array.z == rhs._array.z;
    r &= lhs._array.w == rhs._array.w;

    return r;
  }

  /**
   * @brief Checks if two quaternions are not equal.
   *
   * @param lhs The left-hand side quaternion.
   * @param rhs The right-hand side quaternion.
   * @return True if the quaternions are not equal, false otherwise.
   */
  __host__ __device__ __forceinline__ friend bool operator!=(const quaternionf &lhs, const quaternionf &rhs)
  {
    return !(lhs == rhs);
  }

  /**
  * @brief Multiplication assignment operator for a quaternion and a scalar.
  * 
  * @param r The scalar value.
  * @return The updated quaternion after the multiplication.
  */
  __host__ __device__ __forceinline__ quaternionf &operator*=(const Real r)
  {
    quaternionf ql(*this);

    _array.w *= r;
    _array.x *= r;
    _array.y *= r;
    _array.z *= r;

    return *this;
  }

  /**
    * @brief Define a friend function that multiplies two quaternions and returns the result.
    * 
    * @param lhs The left-hand side quaternion.
    * @param rhs The right-hand side quaternion.
    * @return The resulting quaternion after multiplication.
    */
  __host__ __device__ __forceinline__ friend quaternionf operator*(quaternionf &lhs, quaternionf &rhs)
  {
    quaternionf r(lhs);

    // Multiply the new quaternion object with the right-hand side operand
    r *= rhs;

    return r;
  }

  /**
   * @brief Multiplication operator for a scalar and a quaternion.
   * 
   * @param lhs The scalar value.
   * @param rhs The quaternion value.
   * @return The resulting quaternion after the multiplication.
   */
  __host__ __device__ __forceinline__ friend quaternionf operator*(const Real lhs, const quaternionf &rhs)
  {
    quaternionf r(rhs);

    // Multiply the quaternion by the scalar
    r *= lhs;

    return r;
  }

  /**
   * @brief Multiplies a quaternion by a scalar value.
   * 
   * @param lhs The quaternion to be multiplied.
   * @param rhs The scalar value to multiply by.
   * @return The resulting quaternion.
   */
  __host__ __device__ __forceinline__ friend quaternionf operator*(const quaternionf lhs, const Real &rhs)
  {
    quaternionf r(lhs);

    // Multiply the quaternion by the scalar value
    r *= rhs;

    return r;
  }

  /**
   * @brief Divide the current quaternion by a scalar value.
   * 
   * @param r The scalar value to divide by.
   * @return A reference to the current quaternion after division.
   */
  __host__ __device__ __forceinline__ quaternionf &operator/=(const Real r)
  {
    quaternionf ql(*this);

    // Divide each component of the quaternion by the scalar value.
    _array.w /= r;
    _array.x /= r;
    _array.y /= r;
    _array.z /= r;

    return *this;
  }

  /**
   * @brief Divide a quaternion by a scalar value.
   * 
   * @param lhs The quaternion to be divided.
   * @param rhs The scalar value to divide by.
   * @return The resulting quaternion after division.
   */
  __host__ __device__ __forceinline__ friend quaternionf operator/(const quaternionf lhs, const Real &rhs)
  {
    quaternionf r(lhs);
    r /= rhs;
    return r;
  }

  /**
   * @brief Overloaded subtraction operator for adding two quaternions.
   * 
   * @param lhs The left-hand side quaternion.
   * @param rhs The right-hand side quaternion.
   * @return The result of adding the two quaternions.
   */
  __host__ __device__ __forceinline__ friend quaternionf operator+(const quaternionf &lhs, const quaternionf &rhs)
  {
    quaternionf r(lhs);

    // Add the elements of the right-hand side quaternion to the elements of the copy
    r._array += rhs._array;

    return r;
  }

  /**
   * @brief Overloaded subtraction operator for subtracting two quaternions.
   * 
   * @param lhs The left-hand side quaternion.
   * @param rhs The right-hand side quaternion.
   * @return The result of subtracting the two quaternions.
   */
  __host__ __device__ __forceinline__ friend quaternionf operator-(const quaternionf &lhs, const quaternionf &rhs)
  {
    quaternionf r(lhs);

    // Subtract the elements of the right-hand side quaternion from the elements of the copied quaternion
    r._array -= rhs._array;
    
    return r;
  }

  /**
  * @brief Converts a quaternion to a rotation matrix
  * 
  * @return The rotation matrix
  */
  __host__ __device__ __forceinline__ mat3r toRotationMatrix()
  {
    mat3r res;

    Real tx = Real(2) * this->x();
    Real ty = Real(2) * this->y();
    Real tz = Real(2) * this->z();
    Real twx = tx * this->w();
    Real twy = ty * this->w();
    Real twz = tz * this->w();
    Real txx = tx * this->x();
    Real txy = ty * this->x();
    Real txz = tz * this->x();
    Real tyy = ty * this->y();
    Real tyz = tz * this->y();
    Real tzz = tz * this->z();

    // Calculate the elements of the rotation matrix
    res(0, 0) = Real(1) - (tyy + tzz);
    res(0, 1) = txy - twz;
    res(0, 2) = txz + twy;
    res(1, 0) = txy + twz;
    res(1, 1) = Real(1) - (txx + tzz);
    res(1, 2) = tyz - twx;
    res(2, 0) = txz - twy;
    res(2, 1) = tyz + twx;
    res(2, 2) = Real(1) - (txx + tyy);

    return res;
  }
};

#endif
