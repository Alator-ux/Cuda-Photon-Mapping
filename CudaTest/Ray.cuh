#pragma once
#include "vec3.cuh";

namespace cpm {
    struct Ray {
        cpm::vec3 origin;
        cpm::vec3 direction;
        __device__ Ray() : origin(cpm::vec3(0, 0, 0)), direction(cpm::vec3(0, 0, 0)) {}
        __device__ Ray(const cpm::vec3& origin, const cpm::vec3& dir) {
            this->origin = origin;
            this->direction = cpm::vec3::normalize(direction);
        }
        //__device__ Ray reflect_spherical(const cpm::vec3& from, const cpm::vec3& normal) const;
        __device__ Ray reflect(const cpm::vec3& from, const cpm::vec3& normal) const;
        __device__ Ray reflect(const cpm::vec3& from, const cpm::vec3& normal, float dnd) const;
        __device__ bool refract(const cpm::vec3& from, const cpm::vec3& normal, float refr1, float refr2, Ray& out) const;
        __device__ bool refract(const cpm::vec3& from, const cpm::vec3& normal, float refr1, float refr2,
                                float c1, Ray& out) const;
    };
}