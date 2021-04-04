#pragma once

// common headers
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <curand_kernel.h>

#include "ray.h"
#include "vec3.h"

__device__ inline float degrees_to_radians(float degree)
{
    return degree * 3.141592654f / 180.0f;
}



