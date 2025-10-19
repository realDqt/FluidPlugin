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
// nvVector.h - 2-vector, 3-vector, and 4-vector templates and utilities
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
#ifndef NV_VECTOR_H
#define NV_VECTOR_H

namespace nv
{

    template <class T> class vec2;
    template <class T> class vec3;
    template <class T> class vec4;

    //////////////////////////////////////////////////////////////////////
    //
    // vec2 - template class for 2-tuple vector
    //
    //////////////////////////////////////////////////////////////////////
    template <class T>
    class vec2
    {
        public:

            typedef T value_type;
            int size() const
            {
                return 2;
            }

            ////////////////////////////////////////////////////////
            //
            //  Constructors
            //
            ////////////////////////////////////////////////////////

            // Default/scalar constructor
            vec2(const T &t = T())
            {
                for (int i = 0; i < size(); i++)
                {
                    _array[i] = t;
                }
            }

            // Construct from array
            vec2(const T *tp)
            {
                for (int i = 0; i < size(); i++)
                {
                    _array[i] = tp[i];
                }
            }

            // Construct from explicit values
            vec2(const T v0, const T v1)
            {
                x = v0;
                y = v1;
            }

            explicit vec2(const vec3<T> &u)
            {
                for (int i = 0; i < size(); i++)
                {
                    _array[i] = u._array[i];
                }
            }

            explicit vec2(const vec4<T> &u)
            {
                for (int i = 0; i < size(); i++)
                {
                    _array[i] = u._array[i];
                }
            }

            /**
             * @brief Get the value stored in the array.
             * 
             * @return const T* A pointer to the value stored in the array.
             */
            const T *get_value() const
            {
                return _array;
            }

            vec2<T> &set_value(const T *rhs)
            {
                for (int i = 0; i < size(); i++)
                {
                    _array[i] = rhs[i];
                }

                return *this;
            }

            // indexing operators
            T &operator [](int i)
            {
                return _array[i];
            }

            const T &operator [](int i) const
            {
                return _array[i];
            }

            // type-cast operators
            operator T *()
            {
                return _array;
            }

            operator const T *() const
            {
                return _array;
            }

            ////////////////////////////////////////////////////////
            //
            //  Math operators
            //
            ////////////////////////////////////////////////////////

            // scalar multiply assign
            friend vec2<T> &operator *= (vec2<T> &lhs, T d)
            {
                for (int i = 0; i < lhs.size(); i++)
                {
                    lhs._array[i] *= d;
                }

                return lhs;
            }

            // component-wise vector multiply assign
            friend vec2<T> &operator *= (vec2<T> &lhs, const vec2<T> &rhs)
            {
                for (int i = 0; i < lhs.size(); i++)
                {
                    lhs._array[i] *= rhs[i];
                }

                return lhs;
            }

            // scalar divide assign
            friend vec2<T> &operator /= (vec2<T> &lhs, T d)
            {
                if (d == 0)
                {
                    return lhs;
                }

                for (int i = 0; i < lhs.size(); i++)
                {
                    lhs._array[i] /= d;
                }

                return lhs;
            }

            // component-wise vector divide assign
            friend vec2<T> &operator /= (vec2<T> &lhs, const vec2<T> &rhs)
            {
                for (int i = 0; i < lhs.size(); i++)
                {
                    lhs._array[i] /= rhs._array[i];
                }

                return lhs;
            }

            // component-wise vector add assign
            friend vec2<T> &operator += (vec2<T> &lhs, const vec2<T> &rhs)
            {
                for (int i = 0; i < lhs.size(); i++)
                {
                    lhs._array[i] += rhs._array[i];
                }

                return lhs;
            }

            // component-wise vector subtract assign
            friend vec2<T> &operator -= (vec2<T> &lhs, const vec2<T> &rhs)
            {
                for (int i = 0; i < lhs.size(); i++)
                {
                    lhs._array[i] -= rhs._array[i];
                }

                return lhs;
            }

           /**
             * @brief Overloaded unary negation operator for vec2 class.
             *        Negates each element of the vector and returns a new vector.
             * 
             * @tparam T The type of elements in the vector.
             * @param rhs The vector to be negated.
             * @return The negated vector.
             */
            friend vec2<T> operator - (const vec2<T> &rhs)
            {
                vec2<T> rv;

                for (int i = 0; i < rhs.size(); i++)
                {
                    rv._array[i] = -rhs._array[i];
                }

                return rv;
            }

            /**
             * @brief Overloads the + operator to perform element-wise addition of two vec2 objects.
             * 
             * @param lhs The left-hand side vec2 object.
             * @param rhs The right-hand side vec2 object.
             * @return A new vec2 object that is the sum of the two input vec2 objects.
             */
            friend vec2<T> operator + (const vec2<T> &lhs, const vec2<T> &rhs)
            {
                vec2<T> rt(lhs);
                return rt += rhs;
            }

            /**
             * @brief Overloaded subtraction operator for vec2 class.
             * 
             * @param lhs The left-hand side vec2 object.
             * @param rhs The right-hand side vec2 object.
             * @return The result of subtracting rhs from lhs.
             */
            friend vec2<T> operator - (const vec2<T> &lhs, const vec2<T> &rhs)
            {
                vec2<T> rt(lhs);
                return rt -= rhs;
            }

            /**
             * @brief Multiplies a vector by a scalar.
             *
             * @param lhs The vector to be multiplied.
             * @param rhs The scalar to multiply the vector by.
             * @return The result of multiplying the vector by the scalar.
             */
            friend vec2<T> operator * (const vec2<T> &lhs, T rhs)
            {
                vec2<T> rt(lhs);
                return rt *= rhs;
            }

            /**
             * @brief Multiplies a scalar value by a 2D vector.
             *
             * @param lhs The scalar value to multiply.
             * @param rhs The vector to multiply.
             * @return The result of the multiplication.
             */
            friend vec2<T> operator * (T lhs, const vec2<T> &rhs)
            {
                vec2<T> rt(lhs);
                return rt *= rhs;
            }

            /**
             * @brief Multiplies two vec2 objects element-wise.
             * 
             * @param lhs The first vec2 object.
             * @param rhs The second vec2 object.
             * @return The result of the element-wise multiplication.
             */
            friend vec2<T> operator * (const vec2<T> &lhs, const vec2<T> &rhs)
            {
                vec2<T> rt(lhs);
                return rt *= rhs;
            }

            /**
             * @brief Divide a vector by a scalar.
             *
             * @param lhs The vector to be divided.
             * @param rhs The scalar divisor.
             * @return The resulting vector after division.
             */
            friend vec2<T> operator / (const vec2<T> &lhs, T rhs)
            {
                vec2<T> rt(lhs);
                return rt /= rhs;
            }

            /**
             * @brief Overload of the division operator for component-wise division of vec2 objects.
             * 
             * @param lhs The numerator vec2 object.
             * @param rhs The denominator vec2 object.
             * @return The result of the component-wise division of the two vec2 objects.
             */
            friend vec2<T> operator / (const vec2<T> &lhs, const vec2<T> &rhs)
            {
                vec2<T> rt(lhs);
                return rt /= rhs;
            }

            ////////////////////////////////////////////////////////
            //
            //  Comparison operators
            //
            ////////////////////////////////////////////////////////

            /**
             * @brief Overload of the equality operator for vec2 objects.
             * 
             * @param lhs The first vec2 object.
             * @param rhs The second vec2 object.
             * @return true if the two vec2 objects are equal, false otherwise.
             */
            friend bool operator == (const vec2<T> &lhs, const vec2<T> &rhs)
            {
                bool r = true;

                for (int i = 0; i < lhs.size(); i++)
                {
                    r &= lhs._array[i] == rhs._array[i];
                }

                return r;
            }

            /**
             * @brief Overload the != operator for the vec2 class.
             * 
             * @param lhs The left-hand side vec2.
             * @param rhs The right-hand side vec2.
             * @return True if the two vec2 objects are not equal, false otherwise.
             */
            friend bool operator != (const vec2<T> &lhs, const vec2<T> &rhs)
            {
                bool r = true;

                // Iterate over each element of the vec2
                for (int i = 0; i < lhs.size(); i++)
                {
                    // Check if the corresponding elements are not equal
                    r &= lhs._array[i] != rhs._array[i];
                }

                return r;
            }

            //data intentionally left public to allow vec2.x
            union
            {
                struct
                {
                    T x,y;          // standard names for components
                };
                struct
                {
                    T s,t;          // standard names for components
                };
                T _array[2];     // array access
            };
    };

    //////////////////////////////////////////////////////////////////////
    //
    // vec3 - template class for 3-tuple vector
    //
    //////////////////////////////////////////////////////////////////////
    template <class T>
    class vec3
    {
        public:

            typedef T value_type;
            int size() const
            {
                return 3;
            }

            ////////////////////////////////////////////////////////
            //
            //  Constructors
            //
            ////////////////////////////////////////////////////////

            // Default/scalar constructor
            vec3(const T &t = T())
            {
                for (int i = 0; i < size(); i++)
                {
                    _array[i] = t;
                }
            }

            // Construct from array
            vec3(const T *tp)
            {
                for (int i = 0; i < size(); i++)
                {
                    _array[i] = tp[i];
                }
            }

            // Construct from explicit values
            vec3(const T v0, const T v1, const T v2)
            {
                x = v0;
                y = v1;
                z = v2;
            }

            explicit vec3(const vec4<T> &u)
            {
                for (int i = 0; i < size(); i++)
                {
                    _array[i] = u._array[i];
                }
            }

            explicit vec3(const vec2<T> &u, T v0)
            {
                x = u.x;
                y = u.y;
                z = v0;
            }

            /**
             * @brief Get the value of the array.
             * 
             * @return const T* A pointer to the array.
             */
            const T *get_value() const
            {
                return _array;
            }

            vec3<T> &set_value(const T *rhs)
            {
                for (int i = 0; i < size(); i++)
                {
                    _array[i] = rhs[i];
                }

                return *this;
            }

            // indexing operators
            T &operator [](int i)
            {
                return _array[i];
            }

            const T &operator [](int i) const
            {
                return _array[i];
            }

            // type-cast operators
            operator T *()
            {
                return _array;
            }

            operator const T *() const
            {
                return _array;
            }

            ////////////////////////////////////////////////////////
            //
            //  Math operators
            //
            ////////////////////////////////////////////////////////

            // scalar multiply assign
            friend vec3<T> &operator *= (vec3<T> &lhs, T d)
            {
                for (int i = 0; i < lhs.size(); i++)
                {
                    lhs._array[i] *= d;
                }

                return lhs;
            }

            // component-wise vector multiply assign
            friend vec3<T> &operator *= (vec3<T> &lhs, const vec3<T> &rhs)
            {
                for (int i = 0; i < lhs.size(); i++)
                {
                    lhs._array[i] *= rhs[i];
                }

                return lhs;
            }

            // scalar divide assign
            friend vec3<T> &operator /= (vec3<T> &lhs, T d)
            {
                if (d == 0)
                {
                    return lhs;
                }

                for (int i = 0; i < lhs.size(); i++)
                {
                    lhs._array[i] /= d;
                }

                return lhs;
            }

            // component-wise vector divide assign
            friend vec3<T> &operator /= (vec3<T> &lhs, const vec3<T> &rhs)
            {
                for (int i = 0; i < lhs.size(); i++)
                {
                    lhs._array[i] /= rhs._array[i];
                }

                return lhs;
            }

            // component-wise vector add assign
            friend vec3<T> &operator += (vec3<T> &lhs, const vec3<T> &rhs)
            {
                for (int i = 0; i < lhs.size(); i++)
                {
                    lhs._array[i] += rhs._array[i];
                }

                return lhs;
            }

            // component-wise vector subtract assign
            friend vec3<T> &operator -= (vec3<T> &lhs, const vec3<T> &rhs)
            {
                for (int i = 0; i < lhs.size(); i++)
                {
                    lhs._array[i] -= rhs._array[i];
                }

                return lhs;
            }

            /**
             * Overloaded unary minus operator for negating a vector.
             *
             * @param rhs The vector to be negated.
             * @return The negated vector.
             */
            friend vec3<T> operator - (const vec3<T> &rhs)
            {
                vec3<T> rv;

                // Negate each element of the vector
                for (int i = 0; i < rhs.size(); i++)
                {
                    rv._array[i] = -rhs._array[i];
                }

                return rv;
            }

            /**
             * @brief Overloads the + operator to perform vector addition.
             * 
             * @param lhs The left-hand side vector.
             * @param rhs The right-hand side vector.
             * @return The result of the vector addition.
             */
            friend vec3<T> operator + (const vec3<T> &lhs, const vec3<T> &rhs)
            {
                vec3<T> rt(lhs);
                return rt += rhs;
            }

            /**
             * @brief Overloaded subtraction operator for vec3 class.
             * 
             * @param lhs The left-hand side vec3 object.
             * @param rhs The right-hand side vec3 object.
             * @return The result of the subtraction as a new vec3 object.
             */
            friend vec3<T> operator - (const vec3<T> &lhs, const vec3<T> &rhs)
            {
                vec3<T> rt(lhs);
                return rt -= rhs;
            }

            /**
             * @brief Multiply a vector by a scalar.
             * 
             * @param lhs The vector to be multiplied.
             * @param rhs The scalar to multiply with.
             * @return The resulting vector.
             */
            friend vec3<T> operator * (const vec3<T> &lhs, T rhs)
            {
                vec3<T> rt(lhs);
                return rt *= rhs;
            }

            /**
             * @brief Multiply a scalar value by a vector and return the result.
             * 
             * @param lhs The scalar value to multiply.
             * @param rhs The vector to multiply.
             * @return The resulting vector.
             */
            friend vec3<T> operator * (T lhs, const vec3<T> &rhs)
            {
                vec3<T> rt(lhs);
                return rt *= rhs;
            }

            /**
             * @brief Multiply two 3D vectors element-wise.
             * 
             * @param lhs The left-hand side vector.
             * @param rhs The right-hand side vector.
             * @return The resulting vector after element-wise multiplication.
             */
            friend vec3<T> operator * (const vec3<T> &lhs, const vec3<T> &rhs)
            {
                vec3<T> rt(lhs);
                return rt *= rhs;
            }

            /**
             * @brief Divide a vector by a scalar.
             *
             * @param lhs The vector to be divided.
             * @param rhs The scalar value to divide the vector by.
             * @return The resulting vector after division.
             */
            friend vec3<T> operator / (const vec3<T> &lhs, T rhs)
            {
                vec3<T> rt(lhs);
                return rt /= rhs;
            }

            /**
             * @brief Overloaded division operator for vec3.
             * 
             * @param lhs The dividend vector.
             * @param rhs The divisor vector.
             * @return The resulting vector after division.
             */
            friend vec3<T> operator / (const vec3<T> &lhs, const vec3<T> &rhs)
            {
                vec3<T> rt(lhs);
                return rt /= rhs;
            }

            ////////////////////////////////////////////////////////
            //
            //  Comparison operators
            //
            ////////////////////////////////////////////////////////

            /**
             * @brief Overloaded equality operator for comparing two vec3 objects.
             * 
             * @param lhs The left-hand side vec3 object.
             * @param rhs The right-hand side vec3 object.
             * @return true if the vec3 objects are equal, false otherwise.
             */
            friend bool operator == (const vec3<T> &lhs, const vec3<T> &rhs)
            {
                bool r = true;

                // Iterate over each element of the vec3 objects
                for (int i = 0; i < lhs.size(); i++)
                {
                    r &= lhs._array[i] == rhs._array[i];
                }

                return r;
            }

            /**
             * @brief Overloaded inequality operator for vec3.
             *
             * @param lhs The first vec3 object.
             * @param rhs The second vec3 object.
             * @return True if the two vec3 objects are not equal, false otherwise.
             */
            friend bool operator != (const vec3<T> &lhs, const vec3<T> &rhs)
            {
                bool r = true;

                // Iterate over each element of the vec3 objects
                for (int i = 0; i < lhs.size(); i++)
                {
                    r &= lhs._array[i] != rhs._array[i];
                }

                return r;
            }

            ////////////////////////////////////////////////////////////////////////////////
            //
            // dimension specific operations
            //
            ////////////////////////////////////////////////////////////////////////////////

            /**
             * @brief Calculates the cross product of two 3D vectors.
             *
             * @param lhs The first vector.
             * @param rhs The second vector.
             * @return The cross product of the two vectors.
             */
            friend vec3<T> cross(const vec3<T> &lhs, const vec3<T> &rhs)
            {
                vec3<T> r;
                
                // Calculate the components of the cross product
                r.x = lhs.y * rhs.z - lhs.z * rhs.y;
                r.y = lhs.z * rhs.x - lhs.x * rhs.z;
                r.z = lhs.x * rhs.y - lhs.y * rhs.x;

                return r;
            }

            //data intentionally left public to allow vec2.x
            union
            {
                struct
                {
                    T x, y, z;          // standard names for components
                };
                struct
                {
                    T s, t, r;          // standard names for components
                };
                T _array[3];     // array access
            };
    };

    //////////////////////////////////////////////////////////////////////
    //
    // vec4 - template class for 4-tuple vector
    //
    //////////////////////////////////////////////////////////////////////
    template <class T>
    class vec4
    {
        public:

            typedef T value_type;
            int size() const
            {
                return 4;
            }

            ////////////////////////////////////////////////////////
            //
            //  Constructors
            //
            ////////////////////////////////////////////////////////

            // Default/scalar constructor
            vec4(const T &t = T())
            {
                for (int i = 0; i < size(); i++)
                {
                    _array[i] = t;
                }
            }

            // Construct from array
            vec4(const T *tp)
            {
                for (int i = 0; i < size(); i++)
                {
                    _array[i] = tp[i];
                }
            }

            // Construct from explicit values
            vec4(const T v0, const T v1, const T v2, const T v3)
            {
                x = v0;
                y = v1;
                z = v2;
                w = v3;
            }

            explicit vec4(const vec3<T> &u, T v0)
            {
                x = u.x;
                y = u.y;
                z = u.z;
                w = v0;
            }

            explicit vec4(const vec2<T> &u, T v0, T v1)
            {
                x = u.x;
                y = u.y;
                z = v0;
                w = v1;
            }

            /**
             * @brief Get the value from the array.
             * 
             * @tparam T The type of the value.
             * @return const T* Pointer to the value in the array.
             */
            const T *get_value() const
            {
                return _array;
            }

            vec4<T> &set_value(const T *rhs)
            {
                for (int i = 0; i < size(); i++)
                {
                    _array[i] = rhs[i];
                }

                return *this;
            }

            // indexing operators
            T &operator [](int i)
            {
                return _array[i];
            }

            const T &operator [](int i) const
            {
                return _array[i];
            }

            // type-cast operators
            operator T *()
            {
                return _array;
            }

            operator const T *() const
            {
                return _array;
            }

            ////////////////////////////////////////////////////////
            //
            //  Math operators
            //
            ////////////////////////////////////////////////////////

            // scalar multiply assign
            friend vec4<T> &operator *= (vec4<T> &lhs, T d)
            {
                for (int i = 0; i < lhs.size(); i++)
                {
                    lhs._array[i] *= d;
                }

                return lhs;
            }

            // component-wise vector multiply assign
            friend vec4<T> &operator *= (vec4<T> &lhs, const vec4<T> &rhs)
            {
                for (int i = 0; i < lhs.size(); i++)
                {
                    lhs._array[i] *= rhs[i];
                }

                return lhs;
            }

            // scalar divide assign
            friend vec4<T> &operator /= (vec4<T> &lhs, T d)
            {
                if (d == 0)
                {
                    return lhs;
                }

                for (int i = 0; i < lhs.size(); i++)
                {
                    lhs._array[i] /= d;
                }

                return lhs;
            }

            // component-wise vector divide assign
            friend vec4<T> &operator /= (vec4<T> &lhs, const vec4<T> &rhs)
            {
                for (int i = 0; i < lhs.size(); i++)
                {
                    lhs._array[i] /= rhs._array[i];
                }

                return lhs;
            }

            // component-wise vector add assign
            friend vec4<T> &operator += (vec4<T> &lhs, const vec4<T> &rhs)
            {
                for (int i = 0; i < lhs.size(); i++)
                {
                    lhs._array[i] += rhs._array[i];
                }

                return lhs;
            }

            // component-wise vector subtract assign
            friend vec4<T> &operator -= (vec4<T> &lhs, const vec4<T> &rhs)
            {
                for (int i = 0; i < lhs.size(); i++)
                {
                    lhs._array[i] -= rhs._array[i];
                }

                return lhs;
            }

            /**
             * @brief Overloaded unary negate operator for vec4 class.
             * 
             * @param rhs The right-hand side vector.
             * @return The result of the unary negate operation.
             */
            friend vec4<T> operator - (const vec4<T> &rhs)
            {
                vec4<T> rv;

                // Iterate over each element of the vector and negate it
                for (int i = 0; i < rhs.size(); i++)
                {
                    rv._array[i] = -rhs._array[i];
                }

                return rv;
            }

            /**
             * @brief Overloaded addition operator for vec4 class.
             * 
             * @param lhs The left-hand side vector.
             * @param rhs The right-hand side vector.
             * @return The result of the addition operation.
             */
            friend vec4<T> operator + (const vec4<T> &lhs, const vec4<T> &rhs)
            {
                vec4<T> rt(lhs);
                return rt += rhs;
            }

            /**
             * @brief Overloaded subtraction operator for vec4 class.
             * 
             * @param lhs The left-hand side vec4.
             * @param rhs The right-hand side vec4.
             * @return The resulting vec4 after subtracting rhs from lhs.
             */
            friend vec4<T> operator - (const vec4<T> &lhs, const vec4<T> &rhs)
            {
                vec4<T> rt(lhs);
                return rt -= rhs;
            }

            /**
             * @brief Multiplies a vector by a scalar value.
             *
             * @param lhs The vector to be multiplied.
             * @param rhs The scalar value.
             * @return The resulting vector after multiplication.
             */
            friend vec4<T> operator * (const vec4<T> &lhs, T rhs)
            {
                vec4<T> rt(lhs);
                return rt *= rhs;
            }

            /**
             * @brief Overloaded multiplication operator for scalar and vec4.
             * 
             * @param lhs The scalar value.
             * @param rhs The vector.
             * @return The resulting vector after element-wise scalar multiplication.
             */
            friend vec4<T> operator * (T lhs, const vec4<T> &rhs)
            {
                vec4<T> rt(lhs);
                return rt *= rhs;
            }

            /**
             * @brief Overloaded multiplication operator for vec4.
             * 
             * @param lhs The left-hand side vector.
             * @param rhs The right-hand side vector.
             * @return The resulting vector after element-wise multiplication.
             */
            friend vec4<T> operator * (const vec4<T> &lhs, const vec4<T> &rhs)
            {
                vec4<T> rt(lhs);
                return rt *= rhs;
            }

            /**
             * @brief Overloaded division operator for dividing a vec4 object by a scalar value.
             *
             * @param lhs The vec4 object to be divided.
             * @param rhs The scalar value to divide by.
             * @return A new vec4 object containing the result of the division.
             */
            friend vec4<T> operator / (const vec4<T> &lhs, T rhs)
            {
                vec4<T> rt(lhs);
                return rt /= rhs;
            }

            /**
             * @brief Overload the division operator for vec4 objects.
             * 
             * @param lhs The left-hand side vec4 object.
             * @param rhs The right-hand side vec4 object.
             * @return vec4<T> Returns the result of dividing lhs by rhs.
             */
            friend vec4<T> operator / (const vec4<T> &lhs, const vec4<T> &rhs)
            {
                vec4<T> rt(lhs);
                return rt /= rhs;
            }

            ////////////////////////////////////////////////////////
            //
            //  Comparison operators
            //
            ////////////////////////////////////////////////////////

            /**
             * @brief Overload the equality operator for vec4 objects.
             * 
             * @param lhs The left-hand side vec4 object.
             * @param rhs The right-hand side vec4 object.
             * @return bool Returns true if the vec4 objects are equal, false otherwise.
             */
            friend bool operator == (const vec4<T> &lhs, const vec4<T> &rhs)
            {
                bool r = true;

                // Iterate over each element in the vec4 objects
                for (int i = 0; i < lhs.size(); i++)
                {
                    // Check if the corresponding elements are equal
                    r &= lhs._array[i] == rhs._array[i];
                }

                return r;
            }

            /**
             * @brief Overloaded inequality operator for the vec4 class.
             * 
             * @param lhs The left-hand side vec4 object.
             * @param rhs The right-hand side vec4 object.
             * @return True if the two vec4 objects are not equal, false otherwise.
             */
            friend bool operator != (const vec4<T> &lhs, const vec4<T> &rhs)
            {
                bool r = true;

                // Compare each element of the vec4 objects
                for (int i = 0; i < lhs.size(); i++)
                {
                    r &= lhs._array[i] != rhs._array[i];
                }

                return r;
            }

            //data intentionally left public to allow vec2.x
            union
            {
                struct
                {
                    T x, y, z, w;          // standard names for components
                };
                struct
                {
                    T s, t, r, q;          // standard names for components
                };
                T _array[4];     // array access
            };
    };

    ////////////////////////////////////////////////////////////////////////////////
    //
    // Generic vector operations
    //
    ////////////////////////////////////////////////////////////////////////////////

    /**
     * @brief Compute the dot product of two vectors.
     * 
     * The dot product is calculated by multiplying corresponding elements of the two vectors and summing them up.
     * The vectors must have the same size.
     * 
     * @tparam T The type of the vectors.
     * @param lhs The first vector.
     * @param rhs The second vector.
     * @return The dot product of the two vectors.
     */
    template<class T>
    inline typename T::value_type dot(const T &lhs, const T &rhs)
    {
        typename T::value_type r = 0;
        
        // Iterate over the elements of the vectors
        for (int i = 0; i < lhs.size(); i++)
        {
            // Multiply the corresponding elements of the vectors and add the result to the running sum
            r += lhs._array[i] * rhs._array[i];
        }

        return r;
    }

    /**
     * @brief Calculate the length of the provided vector.
     * 
     * @tparam T The type of the vector.
     * @param vec The vector for which to calculate the length.
     * @return The length of the vector.
     */
    template< class T>
    inline typename T::value_type length(const T &vec)
    {
        typename T::value_type r = 0;

        // Iterate over each element of the vector.
        for (int i = 0; i < vec.size(); i++)
        {
            // Square the element and add it to the result.
            r += vec._array[i]*vec._array[i];
        }

        return typename T::value_type(sqrt(r));
    }

    /**
     * @brief Calculates the squared norm of a vector.
     * 
     * @tparam T The type of the vector.
     * @param vec The vector for which to calculate the squared norm.
     * @return The squared norm of the vector.
     */
    template< class T>
    inline typename T::value_type square_norm(const T &vec)
    {
        typename T::value_type r = 0;

        // Calculate the squared norm by summing the squares of each element in the vector
        for (int i = 0; i < vec.size(); i++)
        {
            r += vec._array[i]*vec._array[i];
        }

        return r;
    }

    /**
     * @brief Normalizes a vector.
     * 
     * @param vec The vector to normalize.
     * @return The normalized vector.
     */
    template< class T>
    inline T normalize(const T &vec)
    {
        typename T::value_type sum(0);
        T r;

        // Calculate the sum of the squares of the vector elements
        for (int i = 0; i < vec.size(); i++)
        {
            sum += vec._array[i] * vec._array[i];
        }

        // Calculate the square root of the sum
        sum = typename T::value_type(sqrt(sum));

        // Normalize the vector elements by dividing each element by the sum
        if (sum > 0)
            for (int i = 0; i < vec.size(); i++)
            {
                r._array[i] = vec._array[i] / sum;
            }

        return r;
    }

    // In VC8 : min and max are already defined by a #define...
#ifdef min
#undef min
#endif
#ifdef max
#undef max
#endif
    /**
     * @brief Returns a new array with the component-wise minimum values between two arrays.
     * 
     * @tparam T The type of the arrays.
     * @param lhs The first array.
     * @param rhs The second array.
     * @return The array with the component-wise minimum values.
     */
    template< class T>
    inline T min(const T &lhs, const T &rhs)
    {
        // Create a new array to store the result
        T rt;

        // Iterate through each element of the arrays
        for (int i = 0; i < lhs.size(); i++)
        {
            // Find the minimum value between the corresponding elements
            rt._array[i] = std::min(lhs._array[i], rhs._array[i]);
        }

        return rt;
    }

    /**
     * @brief Returns a new array with the component-wise maximum values between two arrays.
     * 
     * @tparam T The type of the arrays.
     * @param lhs The first array.
     * @param rhs The second array.
     * @return The array with the component-wise maximum values.
     */
    template< class T>
    inline T max(const T &lhs, const T &rhs)
    {
        // Create a new array to store the result
        T rt;

        // Iterate through each element of the arrays
        for (int i = 0; i < lhs.size(); i++)
        {
            // Find the maximum value between the corresponding elements
            rt._array[i] = std::max(lhs._array[i], rhs._array[i]);
        }

        return rt;
    }


};

#endif
