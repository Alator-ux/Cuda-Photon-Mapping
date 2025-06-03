#pragma once
#include <cuda_runtime.h>

enum class PathType {
    dif_refl = 0, spec_refl, absorption, refr, none
};
class PathOperator
{
    int diffuse_surfs;
    PathType lastPathType;
public:
    /// <summary>
    /// ph — photon map,
    /// rt - ray tracer
    /// </summary>
    __host__ __device__ PathOperator() {
        clear();
    }
    __host__ __device__ void inform(PathType pt) {
        if (pt == PathType::dif_refl) {
            diffuse_surfs++;
        }
        lastPathType = pt;
    }
    /// <summary>
    /// Returns true if caustic map needs to be filled in
    /// </summary>
    __host__ __device__ bool response() const {
        if ((lastPathType == PathType::spec_refl || lastPathType == PathType::refr)
            && diffuse_surfs == 0) {
            return true;
        }
        if (lastPathType == PathType::dif_refl) {
            return false;
        }
        return false;
    }
    __host__ __device__ void clear() {
        diffuse_surfs = 0;
        lastPathType = PathType::none;
    }
};