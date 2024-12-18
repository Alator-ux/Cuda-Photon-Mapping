#pragma once
#include "vec3.cuh";

namespace cpm {
    struct Ray {
        cpm::vec3 origin;
        cpm::vec3 direction;
        __host__ __device__ Ray() : origin(cpm::vec3(0, 0, 0)), direction(cpm::vec3(0, 0, 0)) {}
        __host__ __device__ Ray(const cpm::vec3& origin, const cpm::vec3& direction) {
            this->origin = origin;
            this->direction = cpm::vec3::normalize(direction);
        }
        //__device__ Ray reflect_spherical(const cpm::vec3& from, const cpm::vec3& normal) const;
        __host__ __device__ cpm::Ray reflect(const cpm::vec3& from, const cpm::vec3& normal, float dnd) const {
            cpm::vec3 refl_dir = direction - 2.f * normal * dnd;
            Ray res;
            res.direction = cpm::vec3::normalize(refl_dir);
            res.origin = from + 0.01f * normal;
            return res;
        }
        __host__ __device__ cpm::Ray reflect(const cpm::vec3& from, const cpm::vec3& normal) const {
            auto dnd = cpm::vec3::dot(direction, normal);
            cpm::vec3 refl_dir = direction - 2.f * normal * dnd;
            cpm::Ray res;
            res.direction = cpm::vec3::normalize(refl_dir);
            res.origin = from + 0.01f * normal;
            return res;
        }
        __host__ __device__ bool refract(const cpm::vec3& from, const cpm::vec3& normal, float refr1, float refr2,
            float c1, cpm::Ray& out) const {
            float eta = refr1 / refr2;
            c1 = -c1;
            float w = eta * c1;
            float c2m = (w - eta) * (w + eta);
            if (c2m < -1.f) {
                return false;
            }
            out.direction = eta * direction + (w - sqrt(1.f + c2m)) * normal;
            out.origin = from + 0.01f * -normal;
            return true;
        }
        __host__ __device__ bool refract(const cpm::vec3& from, const cpm::vec3& normal, float refr1, float refr2, cpm::Ray& out) const {
            float eta = refr1 / refr2;
            float c1 = -cpm::vec3::dot(direction, normal);
            float w = eta * c1;
            float c2m = (w - eta) * (w + eta);
            if (c2m < -1.f) {
                return false;
            }
            out.direction = eta * direction + (w - sqrt(1.f + c2m)) * normal;
            out.origin = from + 0.01f * -normal;
            return true;
        }
    };
}