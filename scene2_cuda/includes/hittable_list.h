#pragma once

#include <memory>
#include <vector>

#include "common.h"
#include "hittable.h"


class hittable_list : public hittable
{
public:
	hittable** objects;
	int size;

	__device__ hittable_list() { }
	__device__ hittable_list(hittable** objects, int size) : objects(objects), size(size) { }
	__device__ virtual bool hit(const ray& r, float tMin, float tMax, hitRecord& rec) const override;
};

__device__ bool hittable_list::hit(const ray& r, float tMin, float tMax, hitRecord& rec) const
{
	hitRecord tempRec;
	bool hasHit = false;
	float closest_t = tMax;

	for (int i = 0; i < size; i++)
	{
		if (objects[i]->hit(r, tMin, closest_t, tempRec))
		{
			hasHit = true;
			closest_t = tempRec.t;
			rec = tempRec;
		}
	}

	return hasHit;
}