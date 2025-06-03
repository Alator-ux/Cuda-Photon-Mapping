#pragma once
#include "vec3.cuh";
#include "CudaRandom.cuh"
#include "Defines.cuh"


namespace cpm {
    struct Ray {
        cpm::vec3 origin;
        cpm::vec3 direction;
        __host__ __device__ Ray() : origin(cpm::vec3(0, 0, 0)), direction(cpm::vec3(0, 0, 0)) {}
        __host__ __device__ Ray(const cpm::vec3& origin, const cpm::vec3& direction) {
            this->origin = origin;
            this->direction = cpm::vec3::normalize(direction);
        }
        __device__ Ray reflect_spherical(const cpm::vec3& from, const cpm::vec3& normal, 
            cudaRandomStateT* local_state) const {
            /*float e1 = Random<float>::random();
            float e2 = Random<float>::random();
            float theta = std::pow(std::cos(std::sqrt(e1)), -1.f);
            float phi = M_PI * e2;
            glm::vec3 new_dir;
            float r = glm::length(normal);
            float sin_theta = sin(theta);
            new_dir.x = r * sin_theta * cos(phi);
            new_dir.y = r * sin_theta * sin(phi);
            new_dir.z = r * cos(theta);
            Ray res(from, new_dir);*/
            /*float u = Random<float>::random(-1.f, 1.f);
            float theta = Random<float>::random(0.f, 2.f * M_PI);
            float uc = std::sqrt(1.f - u * u);
            glm::vec3 new_dir(
                uc * std::cos(theta),
                uc * std::sin(theta),
                u
            );*/
            cpm::vec3 new_dir;
            do {
                float x1, x2;
                float sqrx1, sqrx2;
                do {
                    x1 = fmap_to_range(curand_uniform(local_state), -1.f, 1.f);
                    x2 = fmap_to_range(curand_uniform(local_state), -1.f, 1.f);
                    sqrx1 = x1 * x1;
                    sqrx2 = x2 * x2;
                } while (sqrx1 + sqrx2 >= 1);
                float fx1x2 = sqrtf(1.f - sqrx1 - sqrx2);
                new_dir.x = 2.f * x1 * fx1x2;
                new_dir.y = 2.f * x2 * fx1x2;
                new_dir.z = 1.f - 2.f * (sqrx1 + sqrx2);
            } while (cpm::vec3::dot(new_dir, normal) < 0);
            cpm::Ray res;
            res.direction = cpm::vec3::normalize(new_dir);
            res.origin = from + 0.0001f * res.direction;
            return res;
        }
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