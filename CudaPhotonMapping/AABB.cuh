#pragma once
#include "vec3.cuh"
#include "Ray.cuh"
#include "Pair.cuh"

/*
* Source: https://tavianator.com/2022/ray_box_boundary.html
*/
class AABB {
    float corners[6] = {0.f, 0.f, 0.f, 
                        0.f, 0.f, 0.f};
    __host__ __device__ __inline__ 
    inline void update_tmin_tmax(int i, float origin_i, float direction_i, 
                                 float& tmin, float& tmax) const {
        bool sign = signbit(direction_i);
        float bmin = corners[sign * 3 + i];
        float bmax = corners[!sign * 3 + i];
        
        float dmin = (bmin - origin_i) / direction_i;
        float dmax = (bmax - origin_i) / direction_i;

        tmin = fmaxf(dmin, tmin);
        tmax = fminf(dmax, tmax);
    }
public:
    __host__ __device__ AABB() {}
    __host__ __device__ AABB(const cpm::vec3* positions, size_t size) {
        for (int point_i = 0; point_i < 3; point_i++) {
           corners[point_i] = positions[0][point_i];
           corners[point_i + 3] = positions[0][point_i];
        }

        for (size_t i = 1; i < size; i++) {
            auto position = positions[i];
            for (int point_i = 0; point_i < 3; point_i++) {
                if (position[point_i] > corners[point_i + 3]) {
                    corners[point_i + 3] = position[point_i];
                }
                if (position[point_i] < corners[point_i]) {
                    corners[point_i] = position[point_i];
                }
            }
        }
    }
    __host__ __device__ bool intersects_with_ray(const cpm::Ray& ray) const {
        float tmin = 0.0, tmax = INFINITY;

        update_tmin_tmax(0, ray.origin.x, ray.direction.x, tmin, tmax);
        update_tmin_tmax(1, ray.origin.y, ray.direction.y, tmin, tmax);
        update_tmin_tmax(2, ray.origin.z, ray.direction.z, tmin, tmax);   

        return tmin <= tmax;
    }
};