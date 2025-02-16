#pragma once
#include "vec3.cuh"
#include "Ray.cuh"
#include "Pair.cuh"

class AABB {
	cpm::vec3 min, max;
    __host__ __device__ __forceinline__ 
    inline void update_tmin_tmax(float min_t, float max_t, float origin_t, float direction_t, 
                                 float& tmin, float& tmax) const {
        float t1 = (min_t - origin_t) / direction_t;
        float t2 = (max_t - origin_t) / direction_t;
        tmin = fminf(fmaxf(t1, tmin), fmaxf(t2, tmin));
        tmax = fmaxf(fminf(t1, tmax), fminf(t2, tmax));
    }
public:
    __host__ __device__ AABB() : min(0.f), max(0.f) {}
    __host__ __device__ AABB(const cpm::vec3* positions, size_t size) {
        min = positions[0];
        max = positions[0];
        for (size_t i = 1; i < size; i++) {
            auto position = positions[i];
            for (int point_i = 0; point_i < 3; point_i++) {
                if (position[point_i] > max[point_i]) {
                    max[point_i] = position[point_i];
                }
                if (position[point_i] < min[point_i]) {
                    min[point_i] = position[point_i];
                }
            }
        }
    }
    __host__ __device__ bool intersects_with_ray(const cpm::Ray& ray) const {
        float tmin = 0.0, tmax = INFINITY;

        update_tmin_tmax(min.x, max.x, ray.origin.x, ray.direction.x, tmin, tmax);
        update_tmin_tmax(min.y, max.y, ray.origin.y, ray.direction.y, tmin, tmax);
        update_tmin_tmax(min.z, max.z, ray.origin.z, ray.direction.z, tmin, tmax);   

        return tmin <= tmax;
    }
};