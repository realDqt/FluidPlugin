/*
 * Copyright 1993-2013 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

//
// Template math library for common 3D functionality
//
// nvMatrix.h - template matrix code
//
// This code is in part deriver from glh, a cross platform glut helper library.
// The copyright for glh follows this notice.
//
// Copyright (c) NVIDIA Corporation. All rights reserved.
////////////////////////////////////////////////////////////////////////////////

/*
    Copyright (c) 2000 Cass Everitt
    Copyright (c) 2000 NVIDIA Corporation
    All rights reserved.

    Redistribution and use in source and binary forms, with or
    without modification, are permitted provided that the following
    conditions are met:

     * Redistributions of source code must retain the above
       copyright notice, this list of conditions and the following
       disclaimer.

     * Redistributions in binary form must reproduce the above
       copyright notice, this list of conditions and the following
       disclaimer in the documentation and/or other materials
       provided with the distribution.

     * The names of contributors to this software may not be used
       to endorse or promote products derived from this software
       without specific prior written permission.

       THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
       ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
       LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
       FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
       REGENTS OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
       INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
       BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
       LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
       CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
       LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
       ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
       POSSIBILITY OF SUCH DAMAGE.


    Cass Everitt - cass@r3.nu
*/

#ifndef NV_MATRIX_H
#define NV_MATRIX_H

namespace nv
{

    template <class T> class vec2;
    template <class T> class vec3;
    template <class T> class vec4;

    ////////////////////////////////////////////////////////////////////////////////
    //
    //  Matrix
    //
    ////////////////////////////////////////////////////////////////////////////////
    template<class T>
    class matrix4
    {

        public:

            matrix4()
            {
                make_identity();
            }

            matrix4(T t)
            {
                set_value(t);
            }

            matrix4(const T *m)
            {
                set_value(m);
            }

            matrix4(T a00, T a01, T a02, T a03,
                    T a10, T a11, T a12, T a13,
                    T a20, T a21, T a22, T a23,
                    T a30, T a31, T a32, T a33) :
                _11(a00), _12(a01), _13(a02), _14(a03),
                _21(a10), _22(a11), _23(a12), _24(a13),
                _31(a20), _32(a21), _33(a22), _34(a23),
                _41(a30), _42(a31), _43(a32), _44(a33)
            {}

            /**
             * @brief Get the values from the matrix and store them in the given array.
             *
             * @param mp - Pointer to the array where the values will be stored.
             */
            void get_value(T *mp) const
            {
                int c = 0;

                // Iterate over the columns
                for (int j=0; j < 4; j++)
                    // Iterate over the rows
                    for (int i=0; i < 4; i++)
                    {
                        // Store the value in the array
                        mp[c++] = element(i,j);
                    }
            }

            /**
             * @brief Get the values from the matrix and store them in the given array.
             *
             * @return const T* - Pointer to the array containing the values of the matrix.
             */
            const T *get_value() const
            {
                return _array;
            }

            /**
             * @brief Set the value of the elements in a 4x4 matrix.
             * 
             * @param mp Pointer to the memory block containing the values.
             */
            void set_value(T *mp)
            {
                int c = 0;

                // Iterate over each column
                for (int j=0; j < 4; j++)
                    // Iterate over each row
                    for (int i=0; i < 4; i++)
                    {
                        // Set the value of the element at (i, j)
                        element(i,j) = mp[c++];
                    }
            }

            /**
             * @brief Set the value of all elements in a 4x4 matrix to a given value.
             *
             * @param r The value to set all elements to.
             */
            void set_value(T r)
            {
                // Iterate over each row
                for (int i=0; i < 4; i++)
                    // Iterate over each column
                    for (int j=0; j < 4; j++)
                    {
                        // Set the element at (i, j) to the given value
                        element(i,j) = r;
                    }
            }

            /**
             * @brief Set the matrix to an identity matrix.
             */
            void make_identity()
            {
                element(0,0) = 1.0;
                element(0,1) = 0.0;
                element(0,2) = 0.0;
                element(0,3) = 0.0;

                element(1,0) = 0.0;
                element(1,1) = 1.0;
                element(1,2) = 0.0;
                element(1,3) = 0.0;

                element(2,0) = 0.0;
                element(2,1) = 0.0;
                element(2,2) = 1.0;
                element(2,3) = 0.0;

                element(3,0) = 0.0;
                element(3,1) = 0.0;
                element(3,2) = 0.0;
                element(3,3) = 1.0;
            }

            /**
             * @brief Set a uniform scale for the matrix.
             *
             * @param s The scale factor.
             */
            void set_scale(T s)
            {
                element(0,0) = s;
                element(1,1) = s;
                element(2,2) = s;
            }

            /**
             * @brief Set a uniform scale for the matrix.
             *
             * @param s The scale factor.
             */
            void set_scale(const vec3<T> &s)
            {
                // Iterate over each diagonal element of the matrix
                for (int i = 0; i < 3; i++)
                {
                    // Set the diagonal element to the corresponding scale factor
                    element(i,i) = s[i];
                }
            }

            /**
             * @brief Set the translation vector of the matrix.
             *
             * @param t The translation vector.
             */
            void set_translate(const vec3<T> &t)
            {
                // Set the translation vector in the matrix
                for (int i = 0; i < 3; i++)
                {
                    element(i,3) = t[i];
                }
            }
            /**
             * @brief Set the values of a row in the matrix.
             *
             * @param row The index of the row to set.
             * @param values The values to set in the row.
             */
            void set_row(int r, const vec4<T> &t)
            {
                // Iterate over each element in the row
                for (int i = 0; i < 4; i++)
                {
                    // Set the value of the element in the row
                    element(r,i) = t[i];
                }
            }

            /**
             * @brief Set the values of a column in the matrix.
             *
             * @param c The index of the column to set.
             * @param t The values to set in the column.
             */
            void set_column(int c, const vec4<T> &t)
            {
                // Iterate over each row
                for (int i = 0; i < 4; i++)
                {
                    // Set the value of the element at (i, c)
                    element(i,c) = t[i];
                }
            }

            /**
             * @brief Get the specified row of the matrix as a vector
             *
             * @param r The index of the row to get.
             * @return vec4<T> - The vector representing the specified row.
             */
            vec4<T> get_row(int r) const
            {
                vec4<T> v;
                // Iterate over the columns of the matrix
                for (int i = 0; i < 4; i++)
                {
                    // Get the element at the specified row and column
                    v[i] = element(r,i);
                }

                return v;
            }

            /**
             * @brief Get the column of a matrix.
             * 
             * @param c The column index.
             * @return The column vector.
             */
            vec4<T> get_column(int c) const
            {
                vec4<T> v;
                // Iterate over each row of the matrix
                for (int i = 0; i < 4; i++)
                {
                    // Get the element at the current row and column and assign it to the vector
                    v[i] = element(i,c);
                }

                return v;
            }

            /**
             * @brief Calculate the inverse of a 4x4 matrix.
             * 
             * @param m The input matrix.
             * @return The inverse of the input matrix.
             */
            friend matrix4 inverse(const matrix4 &m)
            {
                // Create a new matrix to store the inverse
                matrix4 minv;

                // Create temporary arrays to store the rows of the matrix
                T r1[8], r2[8], r3[8], r4[8];
                T *s[4], *tmprow;

                s[0] = &r1[0];
                s[1] = &r2[0];
                s[2] = &r3[0];
                s[3] = &r4[0];

                register int i,j,p,jj;

                // Copy the elements of the input matrix to the temporary arrays
                for (i=0; i<4; i++)
                {
                    for (j=0; j<4; j++)
                    {
                        s[i][j] = m.element(i,j);

                        // Set the additional elements of the temporary arrays to create an augmented matrix
                        if (i==j)
                        {
                            s[i][j+4] = 1.0;
                        }
                        else
                        {
                            s[i][j+4] = 0.0;
                        }
                    }
                }

                T scp[4];

                // Calculate the scale factors for each row
                for (i=0; i<4; i++)
                {
                    scp[i] = T(fabs(s[i][0]));

                    for (j=1; j<4; j++)
                        if (T(fabs(s[i][j])) > scp[i])
                        {
                            scp[i] = T(fabs(s[i][j]));
                        }

                    if (scp[i] == 0.0)
                    {
                        return minv;    // Return the original matrix if it is singular
                    }
                }

                int pivot_to;
                T scp_max;

                // Perform Gaussian elimination to create an upper triangular matrix
                for (i=0; i<4; i++)
                {
                    // Select pivot row
                    pivot_to = i;
                    scp_max = T(fabs(s[i][i]/scp[i]));

                    // Find out which row should be on top
                    for (p=i+1; p<4; p++)
                        if (T(fabs(s[p][i]/scp[p])) > scp_max)
                        {
                            scp_max = T(fabs(s[p][i]/scp[p]));
                            pivot_to = p;
                        }

                    // Swap the rows if necessary
                    if (pivot_to != i)
                    {
                        tmprow = s[i];
                        s[i] = s[pivot_to];
                        s[pivot_to] = tmprow;
                        T tmpscp;
                        tmpscp = scp[i];
                        scp[i] = scp[pivot_to];
                        scp[pivot_to] = tmpscp;
                    }

                    T mji;

                    // Perform row operations to create zeros below the pivot element
                    for (j=i+1; j<4; j++)
                    {
                        mji = s[j][i]/s[i][i];
                        s[j][i] = 0.0;

                        for (jj=i+1; jj<8; jj++)
                        {
                            s[j][jj] -= mji*s[i][jj];
                        }
                    }
                }

                if (s[3][3] == 0.0)
                {
                    return minv;    // Return the original matrix if it is singular
                }

                //
                // Now we have an upper triangular matrix.
                //
                //  x x x x | y y y y
                //  0 x x x | y y y y
                //  0 0 x x | y y y y
                //  0 0 0 x | y y y y
                //
                //  we'll back substitute to get the inverse
                //
                //  1 0 0 0 | z z z z
                //  0 1 0 0 | z z z z
                //  0 0 1 0 | z z z z
                //  0 0 0 1 | z z z z
                //

                // Perform back substitution to get the inverse matrix
                T mij;

                for (i=3; i>0; i--)
                {
                    for (j=i-1; j > -1; j--)
                    {
                        mij = s[j][i]/s[i][i];

                        for (jj=j+1; jj<8; jj++)
                        {
                            s[j][jj] -= mij*s[i][jj];
                        }
                    }
                }

                for (i=0; i<4; i++)
                    for (j=0; j<4; j++)
                    {
                        minv(i,j) = s[i][j+4] / s[i][i];
                    }

                return minv;
            }

            /**
             * @brief Transpose a matrix.
             *
             * @param m The matrix to transpose.
             * @return The transposed matrix.
             */
            friend matrix4 transpose(const matrix4 &m)
            {
                matrix4 mtrans;
                // Iterate over the rows of the matrix
                for (int i=0; i<4; i++)
                    // Iterate over the columns of the matrix
                    for (int j=0; j<4; j++)
                    {
                        // Set the element at position (i, j) in the transposed matrix
                        mtrans(i,j) = m.element(j,i);
                    }

                return mtrans;
            }

            /**
             * @brief Multiply two matrix4 objects.
             * 
             * @param lhs The first matrix4 object.
             * @param rhs The second matrix4 object.
             * @return A reference to the multiplying the two matrix4 objects.
             */
            matrix4 &operator *= (const matrix4 &rhs)
            {
                // Create a new matrix4 object initialized with zeros
                matrix4 mt(*this);
                set_value(T(0));
                // Iterate over each row of the first matrix
                for (int i=0; i < 4; i++)
                    // Iterate over each column of the second matrix
                    for (int j=0; j < 4; j++)
                        // Iterate over each element of the common dimension
                        for (int c=0; c < 4; c++)
                        {
                            // Multiply and accumulate the corresponding elements
                            element(i,j) += mt(i,c) * rhs(c,j);
                        }

                return *this;
            }

            /**
             * @brief Multiply two matrix4 objects.
             * 
             * @param lhs The first matrix4 object.
             * @param rhs The second matrix4 object.
             * @return The result of multiplying the two matrix4 objects.
             */
            friend matrix4 operator * (const matrix4 &lhs, const matrix4 &rhs)
            {
                // Create a new matrix4 object initialized with zeros
                matrix4 r(T(0));
                // Iterate over each row of the first matrix
                for (int i=0; i < 4; i++)
                    // Iterate over each column of the second matrix
                    for (int j=0; j < 4; j++)
                        // Iterate over each element of the common dimension
                        for (int c=0; c < 4; c++)
                        {
                            // Multiply and accumulate the corresponding elements
                            r.element(i,j) += lhs(i,c) * rhs(c,j);
                        }

                return r;
            }

            /**
             * @brief Multiply a 4-dimensional vector with a matrix.
             *
             * @param src The 4-dimensional vector to be multiplied with the matrix.
             * @return The result of M * src.
             */
            vec4<T> operator *(const vec4<T> &src) const
            {
                vec4<T> r;
                // Perform the matrix multiplication
                for (int i = 0; i < 4; i++)
                    r[i]  = (src[0] * element(i,0) + src[1] * element(i,1) +
                             src[2] * element(i,2) + src[3] * element(i,3));

                return r;
            }

            /**
             * @brief Multiplies a 4-dimensional vector by a 4x4 matrix.
             *
             * @param lhs The vector to be multiplied.
             * @param rhs The matrix to multiply the vector by.
             * @return The result of M * src.
             */
            friend vec4<T> operator *(const vec4<T> &lhs, const matrix4 &rhs)
            {
                vec4<T> r;

                for (int i = 0; i < 4; i++)
                    r[i]  = (lhs[0] * rhs.element(0,i) + lhs[1] * rhs.element(1,i) +
                             lhs[2] * rhs.element(2,i) + lhs[3] * rhs.element(3,i));

                return r;
            }

            T &operator()(int row, int col)
            {
                return element(row,col);
            }

            const T &operator()(int row, int col) const
            {
                return element(row,col);
            }

            T &element(int row, int col)
            {
                return _array[row | (col<<2)];
            }

            const T &element(int row, int col) const
            {
                return _array[row | (col<<2)];
            }

            matrix4 &operator *= (const T &r)
            {
                for (int i = 0; i < 4; ++i)
                {
                    element(0,i) *= r;
                    element(1,i) *= r;
                    element(2,i) *= r;
                    element(3,i) *= r;
                }

                return *this;
            }

            matrix4 &operator += (const matrix4 &mat)
            {
                for (int i = 0; i < 4; ++i)
                {
                    element(0,i) += mat.element(0,i);
                    element(1,i) += mat.element(1,i);
                    element(2,i) += mat.element(2,i);
                    element(3,i) += mat.element(3,i);
                }

                return *this;
            }

            /**
             * @brief Check if two matrix4 objects are equal.
             * 
             * @param lhs The left-hand side matrix4 object.
             * @param rhs The right-hand side matrix4 object.
             * @return True if the two matrix4 objects are equal, false otherwise.
             */
            friend bool operator == (const matrix4 &lhs, const matrix4 &rhs)
            {
                bool r = true;

                for (int i = 0; i < 16; i++)
                {
                    r &= lhs._array[i] == rhs._array[i];
                }

                return r;
            }

            /**
             * @brief Check if two matrix4 objects are not equal.
             *
             * @param lhs The first matrix4 object.
             * @param rhs The second matrix4 object.
             * @return True if the two matrix4 objects are not equal, false otherwise.
             */
            friend bool operator != (const matrix4 &lhs, const matrix4 &rhs)
            {
                bool r = true;
                // Iterate through each element of the matrix arrays
                for (int i = 0; i < 16; i++)
                {
                    r &= lhs._array[i] != rhs._array[i];
                }

                return r;
            }

            union
            {
                struct
                {
                    T _11, _12, _13, _14;   // standard names for components
                    T _21, _22, _23, _24;   // standard names for components
                    T _31, _32, _33, _34;   // standard names for components
                    T _41, _42, _43, _44;   // standard names for components
                };
                T _array[16];     // array access
            };
    };

};

#endif
