#pragma once

#include "common.h"

class material;

struct hitRecord
{
	point3 p;
	vec3 normal;
	material* mat_ptr;
	float t;
	bool front_face;

	__device__ inline void set_face_normal(const ray& r, const vec3& outward_normal)
	{
		front_face = dot(r.direction(), outward_normal) < 0;
		normal = front_face ? outward_normal : -outward_normal;
	}
};

class hittable
{
public:
	__device__ virtual bool hit(const ray& r, float tMin, float tMax, hitRecord& rec) const = 0;
};