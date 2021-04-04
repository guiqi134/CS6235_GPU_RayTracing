#pragma once

#include "common.h"
#include "hittable.h"

class sphere : public hittable
{
public:
	point3 center;
	float radius;
	material* mat_ptr;

	__device__ sphere() { }
	__device__ sphere(point3 c, float r, material* m) : center(c), radius(r), mat_ptr(m) { }
	__device__ virtual bool hit(const ray& r, float tMin, float tMax, hitRecord& rec) const override;
};

__device__ bool sphere::hit(const ray& r, float tMin, float tMax, hitRecord& rec) const
{
	vec3 oc = r.origin() - center;
	float a = r.direction().length_squared();
	float half_b = dot(oc, r.direction());
	float c = oc.length_squared() - radius * radius;
	float discriminant = half_b * half_b - a * c;

	if (discriminant < 0)
		return false;
	float sqrtd = sqrt(discriminant);

	// check whether the smallest root lay in tMin ~ tMax
	float root = (-half_b - sqrtd) / a;
	if (root < tMin || root > tMax)
	{
		root = (-half_b + sqrtd);
		if (root < tMin || root > tMax)
			return false;
	}

	rec.t = root;
	rec.p = r.at(rec.t);
	vec3 outward_normal = (rec.p - center) / radius; // if r < 0, normals points inward.
	rec.set_face_normal(r, outward_normal);
	rec.mat_ptr = mat_ptr;

	return true;
}
