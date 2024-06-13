#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "vec3.h"

namespace cpm {
    struct Photon {
        vec3 pos;
        vec3 power;
        vec3 inc_dir;
        __host__ __device__ Photon() {}
        __host__ __device__ Photon(const Photon& other) {
            this->pos = other.pos;
            this->power = other.power;
            this->inc_dir = other.inc_dir;
        }
        __host__ __device__ Photon(const vec3& pos, const vec3& power, const vec3& inc_dir) {
            this->pos = pos;
            this->power = power;
            this->inc_dir = inc_dir;
        }
    };
}