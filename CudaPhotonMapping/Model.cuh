#pragma once
#include <crt/host_defines.h>
#include <cuda_runtime.h>
#include "ModelConstructInfo.cuh"
#include "Ray.cuh"
#include "Material.cuh"
#include "Light.cuh"
#include "Tuple3.h"
#include "vec3_funcs.cuh"
#include "Pair.cuh"
#include "CudaRandom.cuh"
#include "Ray.cuh"
#include "Defines.cuh"
#include "AABB.cuh"

namespace {
    __device__ int device_model_id_gen = 1;
    int model_id_gen = 1;
    constexpr float model_eps = 1e-6;
    template<typename t>
    t atomicAdd(t* source, t add_value);
}

class Model {
public:
	ModelConstructInfo mci;
    AABB bounding_box;
	int id;

	__host__ __device__ float calculate_triangle_area(const cpm::vec2& v0, const cpm::vec2& v1, const cpm::vec2& v2) const
    {
        return 0.5f * abs((v1.x() - v0.x()) * (v2.y() - v0.y()) - (v2.x() - v0.x()) * (v1.y() - v0.y()));
    }

    __host__ __device__ cpm::vec3 barycentric_coords(const cpm::vec2& st, size_t ii0, size_t ii1, size_t ii2) const {
        auto& st0 = mci.texcoords[ii0];
        auto& st1 = mci.texcoords[ii1];
        auto& st2 = mci.texcoords[ii2];

        return barycentric_coords(st, st0, st1, st2);
    }

    __host__ __device__ cpm::vec3 barycentric_coords(const cpm::vec2& st, cpm::vec2& st0, cpm::vec2& st1, cpm::vec2& st2) const {
        float areaABC = calculate_triangle_area(st0, st1, st2);

        float areaPBC = calculate_triangle_area(st, st1, st2);
        float areaPCA = calculate_triangle_area(st0, st, st2);
        float areaPAB = calculate_triangle_area(st0, st1, st);

        // Calculate the barycentric coordinates using the areas of the sub-triangles
        float barycentricA = areaPBC / areaABC;
        float barycentricB = areaPCA / areaABC;
        float barycentricC = areaPAB / areaABC;

        return cpm::vec3(barycentricA, barycentricB, barycentricC);
    }

    __host__ __device__ bool is_st_in_triangle(const cpm::vec2& st, size_t ii0, size_t ii1, size_t ii2, cpm::vec3& out_uvw) const {
        out_uvw = barycentric_coords(st, ii0, ii1, ii2);
        bool result = out_uvw.x >= 0.0f && out_uvw.y >= 0.0f && out_uvw.z >= 0.0f;
        out_uvw.normalize();
        return result;
    }

    /* floats operation:
    *  86 + some index operations!!!!
    *  86 * 2188 * (100*100) = 1.882 לכנה
    */
    __host__ __device__ bool traingle_intersection(const cpm::Ray& ray, bool in_object,
        const cpm::vec3& v0, const cpm::vec3& v1, const cpm::vec3& v2,
        float& out_ray_parameter, cpm::vec3& out_uvw) const {
        out_ray_parameter = 0.f;
        // compute the plane's normal
        cpm::vec3 first_direction = v1 - v0;
        cpm::vec3 second_direction = v2 - v0;
        // no need to normalize
        cpm::vec3 normal = cpm::vec3::cross(first_direction, second_direction); // Normal of the triangle plane
        float denom = 1.0f / cpm::vec3::dot(normal, normal);

        // Step 1: finding P

        // check if the ray and plane are parallel.
        float NdotRayDirection = cpm::vec3::dot(normal, ray.direction);
        if (fabsf(NdotRayDirection) < model_eps) // almost 0
            return false; // they are parallel so they don't intersect! 

        // compute t (equation 3)
        out_ray_parameter = (cpm::vec3::dot(normal, v0) - cpm::vec3::dot(normal, ray.origin)) / NdotRayDirection;
        // check if the triangle is behind the ray
        if (out_ray_parameter < 0) return false; // the triangle is behind

        // compute the intersection point using equation 1
        cpm::vec3 P = ray.origin + out_ray_parameter * ray.direction;

        // Step 2: inside-outside test
        cpm::vec3 C; // vector perpendicular to triangle's plane

        // edge 0
        first_direction = v1 - v0;
        second_direction = P - v0;
        float w = cpm::vec3::dot(normal, cpm::vec3::cross(first_direction, second_direction));
        if (w < 0) return false; // P is on the right side

        // edge 2
        first_direction = v0 - v2;
        second_direction = P - v2;
        C = cpm::vec3::cross(first_direction, second_direction);
        float v = cpm::vec3::dot(normal, C);
        if (v < 0) return false; // P is on the right side;
        w *= denom;
        v *= denom;
        float u = 1.f - v - w;
        if (u < 0) {
            return false;
        }
        out_uvw.x = u;
        out_uvw.y = v;
        out_uvw.z = w;
        return true; // this ray hits the triangle
    }

public:
	__host__ Model(const Model& other) {
		this->mci = other.mci;
		this->id = other.id;
        this->bounding_box = other.bounding_box;
	}
	__host__ __device__ Model(ModelConstructInfo& mci, int id = -1) : bounding_box(mci.positions, mci.size) {
		this->mci.swap(mci);
		this->id = id;
	}
	__host__ __device__ Model() {
		this->id = -1;
	}
    /* ==============
    *  Equals Section
    *  ============== */
    __host__ __device__ bool equal(const Model& other) const {
        return this->id == other.id;
    }
    __host__ __device__ bool equal(size_t other_id) const {
        return this->id == other_id;
    }
    /* ==============
    *  Getters Section
    *  ============== */
    __host__ __device__ const Material* get_material() const {
        return &mci.material;
    }
    __host__ __device__ int get_id() const {
        return id;
    }
    __host__ __device__ cpm::Tuple3<cpm::vec3, cpm::vec3, cpm::vec3> get_bounding_box() const {
        cpm::vec3 right_upper(mci.positions[0]), left_lower(mci.positions[0]), normal(0.f);
        for (int i = 0; i < mci.size; i++) {
            auto position = mci.positions[i];
            for (int point_i = 0; point_i < 3; point_i++) {
                if (position[point_i] > right_upper[point_i]) {
                    right_upper[point_i] = position[point_i];
                }
                if (position[point_i] < left_lower[point_i]) {
                    left_lower[point_i] = position[point_i];
                }
            }
        }
        for (int i = 0; i < mci.size; i++) {
            normal += mci.normals[i];
        }
        normal /= mci.size;
        normal.normalize();
        return { left_lower, right_upper, normal };
    }
    __host__ __device__ cpm::vec3 get_normal(size_t i) const {
        return mci.normals[i];
    }
    __host__ __device__ void get_normal(size_t ii0, size_t ii1, size_t ii2, cpm::vec2& point, cpm::vec3& normal) {
        if (mci.smooth) {
            auto uvw = barycentric_coords(point, ii0, ii1, ii2);
            auto np = mci.positions[ii0] * uvw.x +
                mci.positions[ii1]* uvw.y + mci.positions[ii2] * uvw.z;
            normal = cpm::interpolate_uvw(mci.positions[ii0], mci.positions[ii1],
                mci.positions[ii2], uvw);
            return;
        }
        normal = mci.normals[ii0];
    }
    __host__ __device__ cpm::vec3 get_smoothed_normal(size_t ii0, size_t ii1, size_t ii2, const cpm::vec3& uvw) {
        cpm::vec3 n0 = mci.normals[ii0];
        cpm::vec3 n1 = mci.normals[ii1];
        cpm::vec3 n2 = mci.normals[ii2];
        n0.mult(uvw.x);
        n1.mult(uvw.y);
        n2.mult(uvw.z);
        return n0.add(n1).add(n2).normalize();
    }
    //__device__ cpm::pair<cpm::vec3, cpm::vec3> Model::get_random_point_with_normal(curandState* state) const 
#ifdef __CUDA_ARCH__
    __device__ cpm::pair<cpm::vec3, cpm::vec3> get_random_point_with_normal(curandState* state) const {
        curandState local_state = *state;

        int start_ind, ind0, ind1, ind2;
        if (mci.type == ModelType::Triangle) {

            start_ind = cpm::fmap_to_range(curand_uniform(&local_state), 0, (mci.size - 3) / 3);
            start_ind *= 3;
            ind0 = start_ind;
            ind1 = start_ind + 1;
            ind2 = start_ind + 2;
        }
        else if (mci.type == ModelType::Quad) {
            start_ind = cpm::fmap_to_range(curand_uniform(&local_state), 0, (mci.size - 4) / 4);
            start_ind *= 4;
            if (curand_uniform(&local_state) < 0.5f) {
                ind0 = start_ind;
                ind1 = start_ind + 1;
                ind2 = start_ind + 3;
            }
            else {
                ind0 = start_ind + 1;
                ind1 = start_ind + 2;
                ind2 = start_ind + 3;
            }
        }
        else {
            printf("Unknown model vertex organization\n");
        }
        cpm::vec3 uvw;
        uvw.x = curand_uniform(&local_state);
        uvw.y = cpm::fmap_to_range(curand_uniform(&local_state), uvw.x, 1.f);
        uvw.z = 1.f - uvw.x - uvw.y;

        cpm::vec3 point = uvw.x * mci.positions[ind0] +
            uvw.y * mci.positions[ind1] +
            uvw.z * mci.positions[ind2];
        cpm::vec3 normal = uvw.x * mci.normals[ind0] +
            uvw.y * mci.normals[ind1] +
            uvw.z * mci.normals[ind2];

        *state = local_state;

        return { point, normal };
    }
#else
    // __host__ cpm::pair<cpm::vec3, cpm::vec3> Model::get_random_point_with_normal() const 
    __host__ cpm::pair<cpm::vec3, cpm::vec3> get_random_point_with_normal(cpm::Random rnd_gen) const {
        int start_ind, ind0, ind1, ind2;
        if (mci.type == ModelType::Triangle) {

            start_ind = rnd_gen.cpurand_int_in_range(0, (mci.size - 3) / 3);
            start_ind *= 3;
            ind0 = start_ind;
            ind1 = start_ind + 1;
            ind2 = start_ind + 2;
        }
        else if (mci.type == ModelType::Quad) {
            start_ind = rnd_gen.cpurand_int_in_range(0, (mci.size - 4) / 4);
            start_ind *= 4;
            if (rnd_gen.cpurand_uniform_in_range(0.f, 1.f) < 0.5f) {
                ind0 = start_ind;
                ind1 = start_ind + 1;
                ind2 = start_ind + 3;
            }
            else {
                ind0 = start_ind + 1;
                ind1 = start_ind + 2;
                ind2 = start_ind + 3;
            }
        }
        else {
            printf("Unknown model vertex organization\n");
        }
        cpm::vec3 uvw;
        uvw.x = rnd_gen.cpurand_uniform();
        uvw.y = rnd_gen.cpurand_uniform_in_range(uvw.x, 1.f);
        uvw.z = 1.f - uvw.x - uvw.y;

        cpm::vec3 point = uvw.x * mci.positions[ind0] +
            uvw.y * mci.positions[ind1] +
            uvw.z * mci.positions[ind2];
        cpm::vec3 normal = uvw.x * mci.normals[ind0] +
            uvw.y * mci.normals[ind1] +
            uvw.z * mci.normals[ind2];

        return { point, normal };
    }
#endif

    /* =============
    *  Other Section
    *  ============= */
    __device__ bool intersection_gpu(const cpm::Ray& ray, bool in_object, float& intersection,
        cpm::vec3& out_uvw) const {
        // Length = 256*3
        extern __shared__ cpm::vec3* normal_ptrs[];
        int thread_ptr_start = threadIdx.x * 3;
        intersection = FLT_MAX;
        size_t i = 0;
        bool intersection_found;
        int primitive_index = 0;
        while (primitive_index < mci.primitives_size) {
            float possible_ray_parameter = 0.f;
            cpm::vec3 possible_uvw;
            if (mci.type == ModelType::Triangle) {
                cpm::vec3 v0 = mci.positions[i];
                cpm::vec3 v1 = mci.positions[i + 1];
                cpm::vec3 v2 = mci.positions[i + 2];
                intersection_found = traingle_intersection(ray, in_object,
                    v0, v1, v2,
                    possible_ray_parameter, possible_uvw);
                if (intersection_found && possible_ray_parameter < intersection) {
                    intersection = possible_ray_parameter;
                    out_uvw = possible_uvw;
                    normal_ptrs[thread_ptr_start]     = mci.normals + i;
                    normal_ptrs[thread_ptr_start + 1] = mci.normals + i + 1;
                    normal_ptrs[thread_ptr_start + 2] = mci.normals + i + 2;
                }
                i += 3;
            }
            else if (mci.type == ModelType::Quad) {
                intersection_found = traingle_intersection(ray, in_object,
                    mci.positions[i], mci.positions[i + 1], mci.positions[i + 3],
                    possible_ray_parameter, possible_uvw);
                if (intersection_found && possible_ray_parameter < intersection) {
                    intersection = possible_ray_parameter;
                    out_uvw = possible_uvw;
                    normal_ptrs[thread_ptr_start]     = mci.normals + i;
                    normal_ptrs[thread_ptr_start + 1] = mci.normals + i + 1;
                    normal_ptrs[thread_ptr_start + 2] = mci.normals + i + 3;
                }
                else {
                    possible_ray_parameter = 0.f;
                    intersection_found = traingle_intersection(ray, in_object,
                        mci.positions[i + 1], mci.positions[i + 2], mci.positions[i + 3],
                        possible_ray_parameter, possible_uvw);
                    if (intersection_found && possible_ray_parameter < intersection) {
                        intersection = possible_ray_parameter;
                        out_uvw = possible_uvw;
                        normal_ptrs[thread_ptr_start]     = mci.normals + i + 1;
                        normal_ptrs[thread_ptr_start + 1] = mci.normals + i + 2;
                        normal_ptrs[thread_ptr_start + 2] = mci.normals + i + 3;
                    }
                }
                i += 4;
            }
            else {
                printf("Unknown model vertex organization\n");
            }
            primitive_index++;
        }
        
        if (intersection == FLT_MAX) {
            return false;
        }
        // TODO Think about access to memory
        /*out_normal = cpm::vec3::normalize(n0 * uvw.x + n1 * uvw.y + n2 * uvw.z);*/

        return true;
    }

    __host__ __device__ __forceinline__ void copy_positions_to_shared(cpm::vec3* shared_pos, cpm::vec3* global_pos, int shared_size, int global_index) const {
        global_index += threadIdx.x;
        for (int local_index = threadIdx.x; (global_index < mci.size) && (local_index < shared_size); local_index += THREADS_NUMBER, global_index += THREADS_NUMBER) {
            shared_pos[local_index] = global_pos[global_index];
        }
    }

    __host__ __device__ bool intersection(const cpm::Ray& ray, bool in_object, float& intersection,
        cpm::vec3& out_normal) const {
        if (!bounding_box.intersects_with_ray(ray)) {
            return false;
        }
        intersection = FLT_MAX;
        size_t ii0 = 0;
        size_t ii1 = 0;
        size_t ii2 = 0;
        cpm::vec3 uvw;
        size_t global_index = 0;
        bool intersection_found;
        int vertices_number = mci.size;
        cpm::vec3* positions = mci.positions;
        ModelType model_type = mci.type;
        while (global_index < vertices_number) {
            float possible_ray_parameter = 0.f;
            cpm::vec3 possible_uvw;
            if (model_type == ModelType::Triangle) {
                cpm::vec3 v0 = positions[global_index];
                cpm::vec3 v1 = positions[global_index + 1];
                cpm::vec3 v2 = positions[global_index + 2];
                intersection_found = traingle_intersection(ray, in_object,
                    v0, v1, v2,
                    possible_ray_parameter, possible_uvw);
                if (intersection_found && possible_ray_parameter < intersection) {
                    intersection = possible_ray_parameter;
                    uvw = possible_uvw;
                    ii0 = global_index;
                    ii1 = global_index + 1;
                    ii2 = global_index + 2;
                }
                global_index += 3;
            }
            else if (model_type == ModelType::Quad) {
                intersection_found = traingle_intersection(ray, in_object,
                    mci.positions[global_index], mci.positions[global_index + 1], mci.positions[global_index + 3],
                    possible_ray_parameter, possible_uvw);
                if (intersection_found && possible_ray_parameter < intersection) {
                    intersection = possible_ray_parameter;
                    uvw = possible_uvw;
                    ii0 = global_index;
                    ii1 = global_index + 1;
                    ii2 = global_index + 3;
                }
                else {
                    possible_ray_parameter = 0.f;
                    intersection_found = traingle_intersection(ray, in_object,
                        mci.positions[global_index + 1], mci.positions[global_index + 2], mci.positions[global_index + 3],
                        possible_ray_parameter, possible_uvw);
                    if (intersection_found && possible_ray_parameter < intersection) {
                        intersection = possible_ray_parameter;
                        uvw = possible_uvw;
                        ii0 = global_index + 1;
                        ii1 = global_index + 2;
                        ii2 = global_index + 3;
                    }
                }
                global_index += 4;
            }
            else {
                printf("Unknown model vertex organization\n");
            }
        }

        // Return false if no intersection was found (ii2 was not updated)
        if (ii2 == 0) {
            return false;
        }
        // TODO Think about access to memory
        cpm::vec3 n0 = mci.normals[ii0];
        cpm::vec3 n1 = mci.normals[ii1];
        cpm::vec3 n2 = mci.normals[ii2];
        n0.mult(uvw.x);
        n1.mult(uvw.y);
        n2.mult(uvw.z);

        out_normal = n0.add(n1).add(n2).normalize();
        /*out_normal = cpm::vec3::normalize(n0 * uvw.x + n1 * uvw.y + n2 * uvw.z);*/
        return true;
    }


    __host__ __device__ bool interpolate_by_st(const cpm::vec2& st, cpm::vec3& out_position, cpm::vec3& out_normal) const {
        cpm::vec3 possible_uvw;
        int primitive_index = 0;
        size_t i = 0;
        bool is_in_triangle;
        while (primitive_index < mci.primitives_size) {
            if (mci.type == ModelType::Triangle) {
                is_in_triangle = is_st_in_triangle(st, i, i + 1, i + 2, possible_uvw);
                if (is_in_triangle) {
                    out_position = cpm::interpolate_uvw(mci.positions[i], mci.positions[i + 1], mci.positions[i + 2], possible_uvw);
                    out_normal = cpm::interpolate_uvw(mci.normals[i], mci.normals[i + 1], mci.normals[i + 2], possible_uvw);
                    return true;
                }
                i += 3;
            }
            else if (mci.type == ModelType::Quad) {
                is_in_triangle = is_st_in_triangle(st, i, i + 1, i + 3, possible_uvw);
                if (is_in_triangle) {
                    out_position = cpm::interpolate_uvw(mci.positions[i], mci.positions[i + 1], mci.positions[i + 3], possible_uvw);
                    out_normal = cpm::interpolate_uvw(mci.normals[i], mci.normals[i + 1], mci.normals[i + 3], possible_uvw);
                    return true;
                }
                else {
                    is_in_triangle = is_st_in_triangle(st, i + 1, i + 2, i + 3, possible_uvw);
                    if (is_in_triangle) {
                        out_position = cpm::interpolate_uvw(mci.positions[i + 1], mci.positions[i + 2], mci.positions[i + 3], possible_uvw);
                        out_normal = cpm::interpolate_uvw(mci.normals[i + 1], mci.normals[i + 2], mci.normals[i + 3], possible_uvw);
                        return true;
                    }
                }
                i += 4;
            }
            else {
                printf("Unknown model vertex organization\n");
            }
            primitive_index++;
        }
        return false;
    }
};


//
//class Model {
//    ModelConstructInfo mci;
//    int id;
//
//    //__host__ __device__ float calculate_triangle_area(const cpm::vec2& v0, const cpm::vec2& v1, const cpm::vec2& v2) const
//    //{
//    //    return 0.5f * abs((v1.x() - v0.x()) * (v2.y() - v0.y()) - (v2.x() - v0.x()) * (v1.y() - v0.y()));
//    //}
//
//    //__host__ __device__ cpm::vec3 barycentric_coords(const cpm::vec2& st, size_t ii0, size_t ii1, size_t ii2) const {
//    //    auto& st0 = mci.texcoords[ii0];
//    //    auto& st1 = mci.texcoords[ii1];
//    //    auto& st2 = mci.texcoords[ii2];
//
//    //    return barycentric_coords(st, st0, st1, st2);
//    //}
//
//    //__host__ __device__ cpm::vec3 barycentric_coords(const cpm::vec2& st, cpm::vec2& st0, cpm::vec2& st1, cpm::vec2& st2) const {
//    //    float areaABC = calculate_triangle_area(st0, st1, st2);
//
//    //    float areaPBC = calculate_triangle_area(st, st1, st2);
//    //    float areaPCA = calculate_triangle_area(st0, st, st2);
//    //    float areaPAB = calculate_triangle_area(st0, st1, st);
//
//    //    // Calculate the barycentric coordinates using the areas of the sub-triangles
//    //    float barycentricA = areaPBC / areaABC;
//    //    float barycentricB = areaPCA / areaABC;
//    //    float barycentricC = areaPAB / areaABC;
//
//    //    return cpm::vec3(barycentricA, barycentricB, barycentricC);
//    //}
//
//    //__host__ __device__ bool is_st_in_triangle(const cpm::vec2& st, size_t ii0, size_t ii1, size_t ii2, cpm::vec3& out_uvw) const {
//    //    out_uvw = barycentric_coords(st, ii0, ii1, ii2);
//    //    bool result = out_uvw.x() >= 0.0f && out_uvw.y() >= 0.0f && out_uvw.z() >= 0.0f;
//    //    out_uvw.normalize();
//    //    return result;
//    //}
//
//    //__host__ __device__ bool traingle_intersection(const cpm::Ray& ray, bool in_object,
//    //    const cpm::vec3& v0, const cpm::vec3& v1, const cpm::vec3& v2,
//    //    float& out_ray_parameter, cpm::vec3& out_uvw) const {
//    //    out_ray_parameter = 0.f;
//    //    // compute the plane's normal
//    //    cpm::vec3 v0v1 = v1 - v0;
//    //    cpm::vec3 v0v2 = v2 - v0;
//    //    // no need to normalize
//    //    cpm::vec3 N = cpm::vec3::cross(v0v1, v0v2); // N
//    //    float denom = cpm::vec3::dot(N, N);
//
//    //    // Step 1: finding P
//
//    //    // check if the ray and plane are parallel.
//    //    float NdotRayDirection = cpm::vec3::dot(N, ray.direction);
//    //    if (fabs(NdotRayDirection) < MODEL_EPS) // almost 0
//    //        return false; // they are parallel so they don't intersect! 
//
//    //    // compute t (equation 3)
//    //    out_ray_parameter = (cpm::vec3::dot(N, v0) - cpm::vec3::dot(N, ray.origin)) / NdotRayDirection;
//    //    // check if the triangle is behind the ray
//    //    if (out_ray_parameter < 0) return false; // the triangle is behind
//
//    //    // compute the intersection point using equation 1
//    //    cpm::vec3 P = ray.origin + out_ray_parameter * ray.direction;
//
//    //    // Step 2: inside-outside test
//    //    cpm::vec3 C; // vector perpendicular to triangle's plane
//
//    //    // edge 0
//    //    cpm::vec3 edge0 = v1 - v0;
//    //    cpm::vec3 vp0 = P - v0;
//    //    C = cpm::vec3::cross(edge0, vp0);
//    //    float w = cpm::vec3::dot(N, C);
//    //    if (w < 0) return false; // P is on the right side
//
//    //    // edge 2
//    //    cpm::vec3 edge2 = v0 - v2;
//    //    cpm::vec3 vp2 = P - v2;
//    //    C = cpm::vec3::cross(edge2, vp2);
//    //    float v = cpm::vec3::dot(N, C);
//    //    if (v < 0) return false; // P is on the right side;
//    //    w /= denom;
//    //    v /= denom;
//    //    float u = 1.f - v - w;
//    //    if (u <= 0) {
//    //        return false;
//    //    }
//    //    out_uvw[0] = u;
//    //    out_uvw[1] = v;
//    //    out_uvw[2] = w;
//    //    return true; // this ray hits the triangle
//    //}
//public:
//    /* ====================
//    *  Constructors Section
//    *  ==================== */
//    __host__ __device__ Model() : id(-1), mci() { }
//    __host__ __device__ Model(const Model& other) {
//        this->id = other.id;
//        this->mci = other.mci;
//    }
//    /*__host__ __device__ Model(const ModelConstructInfo& mci, LightSource* ls = nullptr) {
//        this->mci = mci;
//#ifdef __CUDA_ARCH__
//        this->id = atomicAdd(&device_model_id_gen, 1);
//#else
//        this->id = model_id_gen++;
//#endif
//    }*/
////    /* ==============
////    *  Equals Section
////    *  ============== */
////    __host__ __device__ bool equal(const Model& other) const {
////        return this->id == other.id;
////    }
////    __host__ __device__ bool equal(size_t other_id) const {
////        return this->id == other_id;
////    }
////    /* ==============
////    *  Getters Section
////    *  ============== */
////    __device__ const Material* get_material() const {
////        return &mci.material;
////    }
////    __device__ int get_id() const {
////        return id;
////    }
////    __device__ Tuple3<cpm::vec3> get_bounding_box() const {
////        cpm::vec3 right_upper(mci.positions[0]), left_lower(mci.positions[0]), normal(0.f);
////        for (int i = 0; i < mci.size; i++) {
////            auto position = mci.positions[i];
////            for (int point_i = 0; point_i < 3; point_i++) {
////                if (position[point_i] > right_upper[point_i]) {
////                    right_upper[point_i] = position[point_i];
////                }
////                if (position[point_i] < left_lower[point_i]) {
////                    left_lower[point_i] = position[point_i];
////                }
////            }
////        }
////        for (int i = 0; i < mci.size; i++) {
////            normal += mci.normals[i];
////        }
////        normal /= mci.size;
////        normal.normalize();
////        return { left_lower, right_upper, normal };
////    }
////    __device__ cpm::vec3 get_normal(size_t i) const {
////        return mci.normals[i];
////    }
////    __device__ void get_normal(size_t ii0, size_t ii1, size_t ii2, cpm::vec2& point, cpm::vec3& normal) {
////        if (mci.smooth) {
////            auto uvw = barycentric_coords(point, ii0, ii1, ii2);
////            auto np = mci.positions[ii0] * uvw.x() +
////                mci.positions[ii1]* uvw.y() + mci.positions[ii2] * uvw.z();
////            normal = cpm::interpolate_uvw(mci.positions[ii0], mci.positions[ii1],
////                mci.positions[ii2], uvw);
////            return;
////        }
////        normal = mci.normals[ii0];
////    }
////    //__device__ cpm::pair<cpm::vec3, cpm::vec3> Model::get_random_point_with_normal(curandState* state) const 
////#ifdef __CUDA_ARCH__
////    __device__ cpm::pair<cpm::vec3, cpm::vec3> get_random_point_with_normal(curandState* state) const {
////        curandState local_state = *state;
////
////        int start_ind, ind0, ind1, ind2;
////        if (mci.type == ModelType::Triangle) {
////
////            start_ind = cpm::fmap_to_range(curand_uniform(&local_state), 0, (mci.size - 3) / 3);
////            start_ind *= 3;
////            ind0 = start_ind;
////            ind1 = start_ind + 1;
////            ind2 = start_ind + 2;
////        }
////        else if (mci.type == ModelType::Quad) {
////            start_ind = cpm::fmap_to_range(curand_uniform(&local_state), 0, (mci.size - 4) / 4);
////            start_ind *= 4;
////            if (curand_uniform(&local_state) < 0.5f) {
////                ind0 = start_ind;
////                ind1 = start_ind + 1;
////                ind2 = start_ind + 3;
////            }
////            else {
////                ind0 = start_ind + 1;
////                ind1 = start_ind + 2;
////                ind2 = start_ind + 3;
////            }
////        }
////        else {
////            printf("Unknown model vertex organization\n");
////        }
////        cpm::vec3 uvw;
////        uvw[0] = curand_uniform(&local_state);
////        uvw[1] = cpm::fmap_to_range(curand_uniform(&local_state), uvw.x(), 1.f);
////        uvw[2] = 1.f - uvw.x() - uvw.y();
////
////        cpm::vec3 point = uvw.x() * mci.positions[ind0] +
////            uvw.y() * mci.positions[ind1] +
////            uvw.z() * mci.positions[ind2];
////        cpm::vec3 normal = uvw.x() * mci.normals[ind0] +
////            uvw.y() * mci.normals[ind1] +
////            uvw.z() * mci.normals[ind2];
////
////        *state = local_state;
////
////        return { point, normal };
////    }
////#else
////    // __host__ cpm::pair<cpm::vec3, cpm::vec3> Model::get_random_point_with_normal() const 
////    __host__ cpm::pair<cpm::vec3, cpm::vec3> get_random_point_with_normal(cpm::Random rnd_gen) const {
////        int start_ind, ind0, ind1, ind2;
////        if (mci.type == ModelType::Triangle) {
////
////            start_ind = rnd_gen.cpurand_int_in_range(0, (mci.size - 3) / 3);
////            start_ind *= 3;
////            ind0 = start_ind;
////            ind1 = start_ind + 1;
////            ind2 = start_ind + 2;
////        }
////        else if (mci.type == ModelType::Quad) {
////            start_ind = rnd_gen.cpurand_int_in_range(0, (mci.size - 4) / 4);
////            start_ind *= 4;
////            if (rnd_gen.cpurand_uniform_in_range(0.f, 1.f) < 0.5f) {
////                ind0 = start_ind;
////                ind1 = start_ind + 1;
////                ind2 = start_ind + 3;
////            }
////            else {
////                ind0 = start_ind + 1;
////                ind1 = start_ind + 2;
////                ind2 = start_ind + 3;
////            }
////        }
////        else {
////            printf("Unknown model vertex organization\n");
////        }
////        cpm::vec3 uvw;
////        uvw[0] = rnd_gen.cpurand_uniform();
////        uvw[1] = rnd_gen.cpurand_uniform_in_range(uvw.x(), 1.f);
////        uvw[2] = 1.f - uvw.x() - uvw.y();
////
////        cpm::vec3 point = uvw.x() * mci.positions[ind0] +
////            uvw.y() * mci.positions[ind1] +
////            uvw.z() * mci.positions[ind2];
////        cpm::vec3 normal = uvw.x() * mci.normals[ind0] +
////            uvw.y() * mci.normals[ind1] +
////            uvw.z() * mci.normals[ind2];
////
////        return { point, normal };
////    }
////#endif
////
////    /* =============
////    *  Other Section
////    *  ============= */
////    __host__ __device__ bool intersection(const cpm::Ray& ray, bool in_object, float& intersection,
////        size_t& ii0, size_t& ii1, size_t& ii2,
////        cpm::vec3& out_normal) const {
////        intersection = 0.f;
////        ii0 = ii1 = ii2 = 0;
////        cpm::vec3 uvw;
////        size_t i = 0;
////        bool intersection_found;
////        int primitive_index = 0;
////        while (primitive_index < mci.primitives_size) {
////            float possible_ray_parameter = 0.f;
////            cpm::vec3 possible_uvw;
////            if (mci.type == ModelType::Triangle) {
////                intersection_found = traingle_intersection(ray, in_object,
////                    mci.positions[i], mci.positions[i + 1], mci.positions[i + 2],
////                    possible_ray_parameter, possible_uvw);
////                if (intersection_found && (intersection == 0 || possible_ray_parameter < intersection)) {
////                    intersection = possible_ray_parameter;
////                    uvw = possible_uvw;
////                    ii0 = i;
////                    ii1 = i + 1;
////                    ii2 = i + 2;
////                }
////                i += 3;
////            }
////            else if (mci.type == ModelType::Quad) {
////                intersection_found = traingle_intersection(ray, in_object,
////                    mci.positions[i], mci.positions[i + 1], mci.positions[i + 3],
////                    possible_ray_parameter, possible_uvw);
////                if (intersection_found && (intersection == 0 || possible_ray_parameter < intersection)) {
////                    intersection = possible_ray_parameter;
////                    uvw = possible_uvw;
////                    ii0 = i;
////                    ii1 = i + 1;
////                    ii2 = i + 3;
////                }
////                else {
////                    possible_ray_parameter = 0.f;
////                    intersection_found = traingle_intersection(ray, in_object,
////                        mci.positions[i + 1], mci.positions[i + 2], mci.positions[i + 3],
////                        possible_ray_parameter, possible_uvw);
////                    if (intersection_found && (intersection == 0 || possible_ray_parameter < intersection)) {
////                        intersection = possible_ray_parameter;
////                        uvw = possible_uvw;
////                        ii0 = i + 1;
////                        ii1 = i + 2;
////                        ii2 = i + 3;
////                    }
////                }
////                i += 4;
////            }
////            else {
////                printf("Unknown model vertex organization\n");
////            }
////            primitive_index++;
////        }
////        if (intersection == 0.f) {
////            return false;
////        }
////        auto& n0 = mci.normals[ii0];
////        auto& n1 = mci.normals[ii1];
////        auto& n2 = mci.normals[ii2];
////        out_normal = cpm::vec3::normalize(n0 * uvw.x() + n1 * uvw.y() + n2 * uvw.z());
////        return true;
////    }
////    __host__ __device__ bool interpolate_by_st(const cpm::vec2& st, cpm::vec3& out_position, cpm::vec3& out_normal) const {
////        cpm::vec3 possible_uvw;
////        int primitive_index = 0;
////        size_t i = 0;
////        bool is_in_triangle;
////        while (primitive_index < mci.primitives_size) {
////            if (mci.type == ModelType::Triangle) {
////                is_in_triangle = is_st_in_triangle(st, i, i + 1, i + 2, possible_uvw);
////                if (is_in_triangle) {
////                    out_position = cpm::interpolate_uvw(mci.positions[i], mci.positions[i + 1], mci.positions[i + 2], possible_uvw);
////                    out_normal = cpm::interpolate_uvw(mci.normals[i], mci.normals[i + 1], mci.normals[i + 2], possible_uvw);
////                    return true;
////                }
////                i += 3;
////            }
////            else if (mci.type == ModelType::Quad) {
////                is_in_triangle = is_st_in_triangle(st, i, i + 1, i + 3, possible_uvw);
////                if (is_in_triangle) {
////                    out_position = cpm::interpolate_uvw(mci.positions[i], mci.positions[i + 1], mci.positions[i + 3], possible_uvw);
////                    out_normal = cpm::interpolate_uvw(mci.normals[i], mci.normals[i + 1], mci.normals[i + 3], possible_uvw);
////                    return true;
////                }
////                else {
////                    is_in_triangle = is_st_in_triangle(st, i + 1, i + 2, i + 3, possible_uvw);
////                    if (is_in_triangle) {
////                        out_position = cpm::interpolate_uvw(mci.positions[i + 1], mci.positions[i + 2], mci.positions[i + 3], possible_uvw);
////                        out_normal = cpm::interpolate_uvw(mci.normals[i + 1], mci.normals[i + 2], mci.normals[i + 3], possible_uvw);
////                        return true;
////                    }
////                }
////                i += 4;
////            }
////            else {
////                printf("Unknown model vertex organization\n");
////            }
////            primitive_index++;
////        }
////        return false;
////    }
//};