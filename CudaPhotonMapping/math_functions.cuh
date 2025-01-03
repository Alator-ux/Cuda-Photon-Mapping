#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "vec3.cuh"

namespace cpm {
    #define EXP 2.718f
    __host__ __device__ inline float pow(float n, float p) {
        return powf(n, p);
    }
    __host__ __device__ inline float distance(const vec3& p1, const vec3& p2) {
        float dx = p1.x - p2.x;
        float dy = p1.y - p2.y;
        float dz = p1.z - p2.z;
        return sqrtf(dx * dx + dy * dy + dz * dz);
    }
    __host__ __device__ inline float abs(float value) { // TODO replace
        if (value < 0) {
            return -value;
        }
        return value;
    }

}