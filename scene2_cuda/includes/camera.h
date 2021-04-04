#pragma once

#include "common.h"

class camera
{
private:
    point3 origin;
    point3 lowerLeftCorner;
    vec3 horizontal;
    vec3 vertical;
    vec3 u, v, w;
    float lensRadius;

public:
    __device__ camera(point3 lookFrom, point3 lookAt, vec3 vup, float vfov, float aspectRatio, float aperture, float focusDist) 
    {
        float theta = degrees_to_radians(vfov);
        float h = tan(theta/2);
        float viewportHeight = 2.0f * h;
        float viewportWidth = aspectRatio * viewportHeight;
        float focalLength = 1.0;

        // camera basis
        w = unit_vector(lookFrom - lookAt);
        u = unit_vector(cross(vup, w));
        v = cross(w, u);

        origin = lookFrom;
        horizontal = focusDist * viewportWidth * u;
        vertical = focusDist * viewportHeight * v;
        lowerLeftCorner = origin - horizontal / 2 - vertical / 2 - w * focusDist;
        lensRadius = aperture / 2.0f;
    }

    __device__ ray get_ray(float s, float t, curandState* state) const 
    {
        vec3 rd = lensRadius * random_in_unit_disk(state);
        vec3 offset = u * rd.x() + v * rd.y();

        return ray(origin + offset, lowerLeftCorner + s * horizontal + t * vertical - origin - offset);
    }
};

//class camera {
//public:
//    __device__ camera() {
//        lower_left_corner = vec3(-2.0, -1.0, -1.0);
//        horizontal = vec3(4.0, 0.0, 0.0);
//        vertical = vec3(0.0, 2.0, 0.0);
//        origin = vec3(0.0, 0.0, 0.0);
//    }
//    __device__ ray get_ray(float u, float v, curandState* state) { return ray(origin, lower_left_corner + u * horizontal + v * vertical - origin); }
//
//    vec3 origin;
//    vec3 lower_left_corner;
//    vec3 horizontal;
//    vec3 vertical;
//};