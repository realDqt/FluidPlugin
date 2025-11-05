#pragma once

#include <float.h>
#include <math.h>
#include "general.h"

typedef float Real;

//// const number
#define REAL_PI Real(3.1415926535897932384626433832795029)
#define REAL_2_PI (Real(2.0) * REAL_PI)
#define REAL_HALF_PI (REAL_PI * Real(0.5))
#define REAL_RADS_PER_DEG (REAL_2_PI / Real(360.0))
#define REAL_DEGS_PER_RAD (Real(360.0) / REAL_2_PI)

#define REAL_EPSILON FLT_EPSILON
#define REAL_INFINITY FLT_MAX
#define REAL_ONE 1.0f
#define REAL_ZERO 0.0f
#define REAL_TWO 2.0f
#define REAL_HALF 0.5f
#define REAL_LARGE_FLOAT 1e18f


//// float
FORCE_INLINE Real fabsr(Real x) { return fabsf(x); }
FORCE_INLINE Real cosr(Real x) { return cosf(x); }
FORCE_INLINE Real sinr(Real x) { return sinf(x); }
FORCE_INLINE Real tanr(Real x) { return tanf(x); }
FORCE_INLINE Real acosr(Real x)
{
    if (x < Real(-1))
        x = Real(-1);
    if (x > Real(1))
        x = Real(1);
    return acosf(x);
}
FORCE_INLINE Real asinr(Real x)
{
    if (x < Real(-1))
        x = Real(-1);
    if (x > Real(1))
        x = Real(1);
    return asinf(x);
}
FORCE_INLINE Real atanr(Real x) { return atanf(x); }
FORCE_INLINE Real atan2r(Real x, Real y) { return atan2f(x, y); }
FORCE_INLINE Real expr(Real x) { return expf(x); }
FORCE_INLINE Real logr(Real x) { return logf(x); }
FORCE_INLINE Real powr(Real x, Real y) { return powf(x, y); }
FORCE_INLINE Real fmodr(Real x, Real y) { return fmodf(x, y); }


FORCE_INLINE Real sqrtr(Real y){
    return sqrtf(y);
}

template <typename T>
inline T squared(T v) {
    return v * v;
}

