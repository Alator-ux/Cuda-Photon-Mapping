#pragma once
#include <crt/host_defines.h>
#include <cuda_runtime.h>
#include "vec3.cuh"

namespace cpm {
    __host__ __device__ cpm::vec3 inline interpolate_uvw(const cpm::vec3& v0, const cpm::vec3& v1, const cpm::vec3& v2, const cpm::vec3& uvw) {
        return cpm::vec3(
            v0.x * uvw.x + v1.x * uvw.y + v2.x * uvw.z,
            v0.y * uvw.x + v1.y * uvw.y + v2.y * uvw.z,
            v0.z * uvw.x + v1.z * uvw.y + v2.z * uvw.z
        );
    }
    __host__ __device__ __forceinline__ cpm::vec3 inline interpolate_uvw_with_clone(
        const cpm::vec3& v0, const cpm::vec3& v1, const cpm::vec3& v2, const cpm::vec3& uvw) {
        cpm::vec3 cloned_v0 = v0;
        cpm::vec3 cloned_v1 = v1;
        cpm::vec3 cloned_v2 = v2;
        cloned_v0.mult(uvw.x);
        cloned_v1.mult(uvw.y);
        cloned_v2.mult(uvw.z);

        return cloned_v0.add(cloned_v1).add(cloned_v2).normalize();
    }
   
}