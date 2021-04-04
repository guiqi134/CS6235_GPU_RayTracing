#pragma once

#include "common.h"
#include "hittable.h"


class material
{
public:
    __device__ virtual bool scatter(const ray& r_in, const hitRecord& rec, color& attenuation, ray& scattered, curandState* state) const = 0;
};

class lambertian : public material 
{
public:
    color albedo;

    __device__ lambertian(const color& a) : albedo(a) {}

    __device__ virtual bool scatter(
        const ray& r_in, const hitRecord& rec, color& attenuation, ray& scattered, curandState* state
    ) const override 
    {
        auto scatter_direction = rec.normal + random_unit_vector(state);

        // Catch degenerate scatter direction
        if (scatter_direction.near_zero())
            scatter_direction = rec.normal;

        scattered = ray(rec.p, scatter_direction);
        attenuation = albedo;
        return true;
    }
};

class metal : public material 
{
public:
    color albedo;
    float fuzz;

    __device__ metal(const color& a, float f) : albedo(a), fuzz(f < 1.0f ? f : 1.0f) {}

    __device__ virtual bool scatter(
        const ray& r_in, const hitRecord& rec, color& attenuation, ray& scattered, curandState* state
    ) const override 
    {
        vec3 reflected = reflect(unit_vector(r_in.direction()), rec.normal);
        vec3 v = fuzz * random_in_unit_sphere(state);
        scattered = ray(rec.p, reflected + v);
       /* while (dot(scattered.direction(), rec.normal) <= 0.0f)
            scattered = ray(rec.p, reflected + fuzz * random_in_unit_sphere(state));
        if (dot(scattered.direction(), rec.normal) <= 0.0f)
            scattered = ray(rec.p, reflected - v);*/
        attenuation = albedo;
        return dot(scattered.direction(), rec.normal) > 0.0f;
    }
};

class dielectric : public material
{
public:
    float ir; // Index of refraction

public:
    __device__ dielectric(float index_of_refraction) : ir(index_of_refraction) {}

    __device__ virtual bool scatter(
        const ray& r_in, const hitRecord& rec, color& attenuation, ray& scattered, curandState* state
    ) const override
    {
        attenuation = color(1.0, 1.0, 1.0);
        float refraction_ratio = rec.front_face ? (1.0 / ir) : ir;

        vec3 unit_direction = unit_vector(r_in.direction());
        float cos_theta = fminf(dot(-unit_direction, rec.normal), 1.0);
        float sin_theta = sqrt(1.0 - cos_theta * cos_theta);

        bool cannot_refract = refraction_ratio * sin_theta > 1.0;
        vec3 direction;
        if (cannot_refract || reflectance(cos_theta, refraction_ratio) > random_float(state))
            direction = reflect(unit_direction, rec.normal);
        else
            direction = refract(unit_direction, rec.normal, refraction_ratio);

        scattered = ray(rec.p, direction);
        return true;
    }

private:
    __device__ static float reflectance(float cosine, float ref_idx)
    {
        // Use Schlick's approximation for reflectance.
        float r0 = (1.0f - ref_idx) / (1.0f + ref_idx);
        r0 = r0 * r0;
        return r0 + (1.0f - r0) * pow(1.0f - cosine, 5.0f);
    }
};