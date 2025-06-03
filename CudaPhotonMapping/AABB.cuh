#pragma once
#include <crt/host_defines.h>
#include <cuda_runtime.h>
#include "vec3.cuh"
#include "Ray.cuh"
#include "Pair.cuh"

/*
* Source: https://tavianator.com/2022/ray_box_boundary.html
*/
class AABB {
public:
    float corners[6] = {FLT_MAX,  FLT_MAX,  FLT_MAX,   // min
                        -FLT_MAX, -FLT_MAX, -FLT_MAX}; // max
protected:
    int blocked = 0; // for atomic operations

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
    __host__ __device__ AABB(const AABB& other) {
        for (int i = 0; i < 6; i++) {
            corners[i] = other.corners[i];
        }
    }
    __host__ __device__ AABB(const cpm::vec3* positions, size_t size) {
        fill(positions, size);
    }
    __host__ __device__ AABB(const cpm::vec3* positions, size_t size, bool calculate) {
        if (calculate) {
            fill(positions, size);
        }
    }
    __host__ __device__ void fill(const cpm::vec3* positions, size_t size) {
        cpm::vec3 position = positions[0];
        add_first(position);

        for (size_t i = 1; i < size; i++) {
            position = positions[i];
            add(position);
        }
    }
    __host__ __device__ void add_first(const cpm::vec3& position) {
        for (int point_i = 0; point_i < 3; point_i++) {
            corners[point_i] = position[point_i];
            corners[point_i + 3] = position[point_i];
        }
    }
    __host__ __device__ void add(const cpm::vec3& position) {
        for (int point_i = 0; point_i < 3; point_i++) {
            if (position[point_i] > corners[point_i + 3]) {
                corners[point_i + 3] = position[point_i];
            }
            if (position[point_i] < corners[point_i]) {
                corners[point_i] = position[point_i];
            }
        }
    }
    __host__ __device__ void add(const AABB& other) {
        for (int point_i = 0; point_i < 3; point_i++) {
            if (other.corners[point_i + 3] > corners[point_i + 3]) {
                corners[point_i + 3] = other.corners[point_i + 3];
            }
            if (other.corners[point_i] < corners[point_i]) {
                corners[point_i] = other.corners[point_i];
            }
        }
    }
    __host__ __device__ void add(const AABB* other) {
        const float* other_corners = other->corners;
        for (int point_i = 0; point_i < 3; point_i++) {
            if (other_corners[point_i + 3] > corners[point_i + 3]) {
                corners[point_i + 3] = other_corners[point_i + 3];
            }
            if (other_corners[point_i] < corners[point_i]) {
                corners[point_i] = other_corners[point_i];
            }
        }
    }
    __device__ void atomicAdd(const AABB& new_aabb) {
        while (atomicCAS(&blocked, 0, 1) != 0) {
            add(new_aabb);
            atomicExch(&blocked, 0);
        }
    }
    __host__ __device__ AABB add_const(const AABB& other) {
        AABB new_aabb(other);
        new_aabb.add(this);
        return new_aabb;
    }
    __host__ __device__ bool intersects_with_ray(const cpm::Ray& ray) const {
        float tmin = 0.0, tmax = INFINITY;

        update_tmin_tmax(0, ray.origin.x, ray.direction.x, tmin, tmax);
        update_tmin_tmax(1, ray.origin.y, ray.direction.y, tmin, tmax);
        update_tmin_tmax(2, ray.origin.z, ray.direction.z, tmin, tmax);   

        return tmin <= tmax;
    }
    __host__ __device__ float length_diff_by_side(int side, const AABB& other) {
        return fabsf(corners[side] - other.corners[side]);
    }
    __host__ __device__ float length_by_axis(int axis) const {
        return fabsf(corners[axis + 3] - corners[axis]);
    }
    __host__ __device__ float median_by_axis(int axis) {
        return (corners[axis + 3] + corners[axis]) * 0.5f; //???
        // return corners[axis + 3] + corners[axis] * 0.5f;
    }
    __host__ __device__ float get_side_value(int side) {
        return corners[side];
    }
    __host__ __device__ void set_side_value(int side, float value) {
        corners[side] = value;
    }
    __host__ __device__ void print() {
        printf("Min: %f %f %f; Max: %f %f %f\n", corners[0], corners[1], corners[2], corners[3], corners[4], corners[5]);
    }
};