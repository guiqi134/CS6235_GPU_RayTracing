#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <curand_kernel.h>

#include <iostream>

//using std::sqrt;
//using std::fabs;

// utility functions
__device__ inline float random_float(curandState* state) { return curand_uniform(state); } // [0, 1]
__device__ inline float random_float(curandState* state, float min, float max) { return min + (max - min) * random_float(state); } // (min, max]
__device__ inline int random_int(curandState* state, int min, int max) { return static_cast<int>(random_float(state, min, max + 1)); }// [min, max]

class vec3
{
public:
    float e[3]; // stand for (x, y, z)

    __host__ __device__ vec3() : e{ 0, 0, 0 } { } // default constr.
    __host__ __device__ vec3(float e0, float e1, float e2) : e{ e0, e1, e2 } { } // param. constr.

    // methods
    __host__ __device__ float x() const { return e[0]; }
    __host__ __device__ float y() const { return e[1]; }
    __host__ __device__ float z() const { return e[2]; }

    __host__ __device__ float length() const
    {
        return sqrtf(length_squared());
    }

    __host__ __device__ float length_squared() const
    {
        return e[0] * e[0] + e[1] * e[1] + e[2] * e[2];
    }

    // Return true if the vector is close to zero in all dimensions.
    __host__ __device__ bool near_zero() const
    {
        const auto s = 1e-8;
        return (fabsf(e[0]) < s) && (fabsf(e[1]) < s) && (fabsf(e[2]) < s);
    }

    __device__ inline static vec3 random(curandState* state)
    {
        return vec3(random_float(state), random_float(state), random_float(state));
    }

    __device__ inline static vec3 random(curandState* state, float min, float max)
    {
        return vec3(random_float(state, min, max), random_float(state, min, max), random_float(state, min, max));
    } 

    // operator overload
    __host__ __device__ vec3 operator-() const { return vec3(-e[0], -e[1], -e[2]); }
    __host__ __device__ float operator[](int i) const { return e[i]; }
    __host__ __device__ float& operator[](int i) { return e[i]; }

    __host__ __device__ vec3& operator+=(const vec3& v)
    {
        e[0] += v.e[0];
        e[1] += v.e[1];
        e[2] += v.e[2];
        return *this;
    }

    __host__ __device__ vec3& operator*=(const float t)
    {
        e[0] *= t;
        e[1] *= t;
        e[2] *= t;
        return *this;
    }

    __host__ __device__ vec3& operator*=(const vec3& v)
    {
        e[0] *= v[0];
        e[1] *= v[1];
        e[2] *= v[2];
        return *this;
    }

    __host__ __device__ vec3& operator/=(const float t)
    {
        return *this *= 1 / t;
    }
};

// Type aliases for vec3
using point3 = vec3;
using color = vec3;

// vec3 utility functions: for binary operators
// << operator only for CPU side 
inline std::ostream& operator<<(std::ostream& out, const vec3& v)
{
    return out << v.e[0] << ' ' << v.e[1] << ' ' << v.e[2];
}

__host__ __device__ inline vec3 operator+(const vec3& u, const vec3& v)
{
    return vec3(u.e[0] + v.e[0], u.e[1] + v.e[1], u.e[2] + v.e[2]);
}

__host__ __device__ inline vec3 operator-(const vec3& u, const vec3& v)
{
    return vec3(u.e[0] - v.e[0], u.e[1] - v.e[1], u.e[2] - v.e[2]);
}

__host__ __device__ inline vec3 operator*(const vec3& u, const vec3& v)
{
    return vec3(u.e[0] * v.e[0], u.e[1] * v.e[1], u.e[2] * v.e[2]);
}

__host__ __device__ inline vec3 operator*(float t, const vec3& u)
{
    return vec3(u.e[0] * t, u.e[1] * t, u.e[2] * t);
}

__host__ __device__ inline vec3 operator*(const vec3& u, float t)
{
    return t * u;
}

__host__ __device__ inline vec3 operator/(const vec3& u, float t)
{
    return (1 / t) * u;
}

__host__ __device__ inline float dot(const vec3& u, const vec3& v)
{
    return u.e[0] * v.e[0] + u.e[1] * v.e[1] + u.e[2] * v.e[2];
}

__host__ __device__ inline vec3 cross(const vec3& u, const vec3& v)
{
    return vec3(u.e[1] * v.e[2] - u.e[2] * v.e[1],
        u.e[2] * v.e[0] - u.e[0] * v.e[2],
        u.e[0] * v.e[1] - u.e[1] * v.e[0]);
}

__host__ __device__ inline vec3 unit_vector(vec3 v)
{
    return v / v.length();
}


// Diffuse Reflection.
// Rejection sampling method: Pick a random point in the unit cube,
// and check whether it lays inside the sphere
__device__ vec3 random_in_unit_sphere(curandState* state)
{
    while (true)
    {
        auto p = vec3::random(state, -1.0f, 1.0f);
        if (p.length_squared() >= 1.0f) continue;
        return p;
    }
}

// True Lambertian Reflection: sample points on the unit sphere
__device__ vec3 random_unit_vector(curandState* state)
{
    return unit_vector(random_in_unit_sphere(state));
}

// Adopting Lambertian diffuse: reflect in a hemisphere
__device__ vec3 random_in_hemisphere(const vec3& normal, curandState* state)
{
    vec3 in_unit_sphere = random_in_unit_sphere(state);
    if (dot(in_unit_sphere, normal) > 0.0)
        return in_unit_sphere;
    else
        return -in_unit_sphere;
}

// Lens for depth of field.
__device__ vec3 random_in_unit_disk(curandState* state)
{
    while (true)
    {
        auto p = vec3(random_float(state, -1, 1), random_float(state, -1, 1), 0);
        if (p.length_squared() >= 1) continue;
        return p;
    }
}

// Mirror Reflection
__device__ vec3 reflect(const vec3& v, const vec3& n)
{
    return v - 2 * dot(v, n) * n;
}

 // Refraction
__device__ vec3 refract(const vec3& uv, const vec3& n, double etai_over_etat)
{
    float cos_theta = fminf(dot(-uv, n), 1.0);
    vec3 r_out_perp = etai_over_etat * (uv + cos_theta * n); // rafraction ray perpendicular to n
    vec3 r_out_parallel = -sqrtf(fabsf(1.0 - r_out_perp.length_squared())) * n; // rafraction ray parallel to n
    return r_out_perp + r_out_parallel;
}