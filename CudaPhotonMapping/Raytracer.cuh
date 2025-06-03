#pragma once
#include "CudaUtils.cuh"
#include "Scene.cuh"
#include "MediumManager.cuh"
#include "DeepLookStack.cuh"
#include "RaytracePlanner.cuh"
#include "PhotonMaxHeap.cuh"
#include "PhotonGrid.cuh"
#include <math_constants.h>

namespace {
    using namespace std;
}

class Raytracer {
public:
    struct IntersectionInfo {
        cpm::vec3 normal, intersection_point;
        Model* incident_model;
        IntersectionInfo() : normal(), intersection_point(), incident_model(nullptr) {}
    };
private:   
    Scene scene;
    RaytracePlanner ray_planner;
    PhotonMaxHeap* heaps;
    PhotonGrid diffuse_grid;
    PhotonGrid specular_grid;

    __device__ bool find_intersection_v2(const cpm::Ray& ray, bool reverse_normal, 
        Model*& out_incident_model, cpm::vec3& out_normal, cpm::vec3& out_intersection_point) {
        // Length = 256*3
        extern __shared__ cpm::vec3* normal_ptrs[];
        int thread_ptr_start = threadIdx.x * 3;

        float intersection = 0.f;
        out_incident_model = nullptr;
        cpm::vec3 uvw;

        int models_number = scene.models_number;
        Model* models = scene.models;
        for (int i = 0; i < models_number; i++) {
            Model* model = models + i;

            float temp_inter;
            cpm::vec3 tuvw;
            bool succ = model->intersection_gpu(ray, false, temp_inter, tuvw);
            if (succ && (intersection == 0.f || temp_inter < intersection)) {
                intersection = temp_inter;
                uvw = tuvw;
                out_incident_model = model;
            }
        }
        if (out_incident_model == nullptr) {
            return false;
        }
        out_intersection_point = (ray.direction * intersection).add(ray.origin);

        out_normal = cpm::interpolate_uvw_with_clone(
            *normal_ptrs[thread_ptr_start], *normal_ptrs[thread_ptr_start + 1], *normal_ptrs[thread_ptr_start + 2], uvw);
        if (reverse_normal && cpm::vec3::dot(ray.direction, out_normal) > 0) {
            out_normal.mult(-1.f);
        }
        return true;
    }

    __host__ __device__ bool find_intersection(const cpm::Ray& ray, bool reverse_normal,
        Model*& out_incident_model, cpm::vec3& out_normal, cpm::vec3& out_intersection_point) {
        float intersection = 0.f;
        out_incident_model = nullptr;

        int models_number = scene.models_number;
        Model* models = scene.models;
        for (int i = 0; i < models_number; i++) {
            Model* model = models + i;

            float temp_inter;
            cpm::vec3 tnormal;
            bool succ = model->intersection(ray, false, temp_inter, tnormal);
            if (succ && (intersection == 0.f || temp_inter < intersection)) {
                intersection = temp_inter;
                out_incident_model = model;
                out_normal = tnormal;
            }
        }
        if (out_incident_model == nullptr) {
            return false;
        }
        out_intersection_point = (ray.direction * intersection).add(ray.origin);
        if (reverse_normal && cpm::vec3::dot(ray.direction, out_normal) > 0) {
            out_normal.mult(-1.f);
        }
        return true;
    }

    __host__ __device__ cpm::vec3 HDR(cpm::vec3 rgb) {
        constexpr float exposure = 0.6f;
        constexpr float gamma = 2.2f;
        float t = pow(exposure, -1);
        cpm::vec3 exp_rgb(expf(-t * rgb.x), expf(-t * rgb.y), expf(-t * rgb.z));
        cpm::vec3 mapped = cpm::vec3(1.0) - exp_rgb;
        constexpr float r_gamma(1.0f / gamma);
        mapped.x = powf(mapped.x, r_gamma);
        mapped.y = powf(mapped.y, r_gamma);
        mapped.z = powf(mapped.z, r_gamma);

        return mapped;
    }


    __host__ __device__ cpm::vec3 get_color_from_photon_grid(PhotonMaxHeapItem* heap_data, uint heap_size,
        cpm::vec3* photon_directions, cpm::vec3* photon_powers,
        const cpm::vec3 normal,
        uint array_idx, uint array_cap) {
        constexpr float alpha = 1.818f;
        constexpr float beta = 1.953f;
        constexpr int straight_coef = 1;
        cpm::vec3 res(0.f);
        if (heap_size == 0) { return res; }
        float r, filter_r;
        filter_r = r = heap_data[PHOTON_HEAP_OFFSET(array_idx, array_cap, 0)].distance;
        filter_r *= filter_r;
        for (int i = 0; i < heap_size; i++) {
            auto elem = heap_data[PHOTON_HEAP_OFFSET(array_idx, array_cap, i)];
            auto photon_direction = photon_directions[elem.idx];
            float cosNL = cpm::vec3::dot(normal, -photon_direction);
            if (cosNL > 0) {
                float distance = elem.distance;
                float filter = alpha * (1.f - (1.f - expf(-beta * distance * distance / (2 * filter_r))) /
                    (1.f - expf(-beta)));
                //float filter = 1;
                res += photon_powers[elem.idx] * filter;
            }
        }

        float common_coef = (1.f / (CUDART_PI_F * (r * r) * straight_coef));
        res *= common_coef;
        return res;
    }

    __host__ __device__ cpm::vec3 render_trace(cpm::Ray current_ray, bool in_object, uint stack_id, uint total_pixels = 1) {
        int new_depth = 0;
        int current_depth = -1;
        cpm::vec3 accumulated_color(0.f);
        cpm::vec3 new_ray_coef(1.f);

        bool max_medium_depth = false;
        while ((current_depth < GlobalParams::max_depth() && current_depth != new_depth)
                || ray_planner.pop_refraction(stack_id, max_medium_depth, current_ray, new_depth, new_ray_coef, in_object)) {

            current_depth = new_depth;
            
            cpm::vec3 normal, inter_p;
            Model* imodel;
            if (!find_intersection(current_ray, in_object, imodel, normal, inter_p)) {
                continue;
            }

            cpm::vec3 current_color(0.f);
            Material mat = *imodel->get_material();

            if (!in_object && !mat.diffuse.is_zero()) {
                cpm::vec3 re(0.f);
                //int important_ls = 0;
                //cpm::vec3 tnormal, tinter_p;
                //Model* timodel;
                //cpm::Ray tray;

                //int ls_number = scene.light_sources_number;
                //LightSource* light_sources = scene.light_sources;
                //for (int i = 0; i < ls_number; i++) {
                //    const LightSource* ls = light_sources + i;
                //    tray.origin = ls->position;
                //    tray.direction = cpm::vec3::normalize(inter_p - ls->position); // point <- ls
                //    tray.origin += tray.direction * 0.001f;
                //    if (find_intersection(tray, in_object, timodel, tnormal, tinter_p) && inter_p.equal(tinter_p)) {
                //        important_ls++;
                //        re += max(cpm::vec3::dot(normal, -tray.direction), 0.f);
                //    }
                //}

                //re /= important_ls == 0 ? 1 : important_ls;
                auto di_op = mat.diffuse * mat.opaque;

                /*re *= di_op;
                current_color += re;*/

                diffuse_grid.find_nearests(inter_p, 0.1f, GlobalParams::global_photon_num(), heaps[stack_id], stack_id, total_pixels);
                uint heap_size = heaps[stack_id].get_size();
                PhotonMaxHeapItem* heap_data = photon_heap_data();
                cpm::vec3* photon_directions = diffuse_grid.get_photon_directions();
                cpm::vec3* photon_powers = diffuse_grid.get_photon_powers();
                re = get_color_from_photon_grid(heap_data, heap_size, photon_directions, photon_powers, normal, stack_id, total_pixels);
                current_color += re * di_op * 4;

            }

            current_color += mat.emission;

            {
                if (specular_grid.find_nearests(inter_p, 0.2f, GlobalParams::caustic_photon_num(), heaps[stack_id], stack_id, total_pixels)) {
                    uint heap_size = heaps[stack_id].get_size();
                    PhotonMaxHeapItem* heap_data = photon_heap_data();
                    cpm::vec3* photon_directions = diffuse_grid.get_photon_directions();
                    cpm::vec3* photon_powers = diffuse_grid.get_photon_powers();
                    cpm::vec3 temp = get_color_from_photon_grid(heap_data, heap_size, photon_directions, photon_powers, normal, stack_id, total_pixels);
                    current_color += temp * mat.opaque;
                }
            }

            current_color.clamp_min(1.f);
            accumulated_color += current_color * new_ray_coef;

            if (mat.opaque < 1.f && current_depth != GlobalParams::max_depth() - 1) {
                int model_id = imodel->get_id();
                
                cpm::Tuple3<float, float, bool> prev_new_refr_ind = ray_planner.get_refractive_indices(
                    stack_id, model_id, mat.refr_index);
                cpm::Ray nray;
                bool succ = current_ray.refract(inter_p, normal,
                    prev_new_refr_ind.item1, prev_new_refr_ind.item2, nray);
                if (succ) {
                    ray_planner.push_refraction(stack_id, max_medium_depth,
                        nray, current_depth, new_ray_coef,
                        model_id, mat.refr_index, mat.opaque, prev_new_refr_ind.item3);
                }
            }

            if (!in_object && !mat.specular.is_zero() && current_depth != GlobalParams::max_depth() - 1) {
                cpm::Ray nray = current_ray.reflect(inter_p, normal);
                float coef = powf(fmaxf(cpm::vec3::dot(nray.direction, -current_ray.direction), 0.0f), mat.shininess);

                new_ray_coef *= mat.specular * mat.opaque * coef;
                current_ray = nray;
                new_depth = current_depth + 1;
            }
        }
        
        accumulated_color = HDR(accumulated_color);
        accumulated_color.clamp_min(1.f);
        return accumulated_color;
    }

   public:
  
    __device__ void render_gpu(uchar3* canvas, int max_working_threads, int width, int height) {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;
        int local_id = threadIdx.y * blockDim.x + threadIdx.x;
        if (x >= width || y >= height) return;

        int id = y * width + x;

        /* ^
        *  | => 60 fps to 46 fps
        */

        /*int id = blockIdx.x * blockDim.x + threadIdx.x;
        if (id >= width * height) {
            return;
        }
        int x = id % width;
        int y = id / width;
        
        int local_id = threadIdx.x;*/

        int total_pixels = width * height;
        for (int i = id; i < total_pixels; i+=max_working_threads) {
            x = i % width;
            y = i / width;
            cpm::Ray ray(scene.camera.position, scene.camera.generate_ray_direction(x, y));
            
            cpm::vec3 pixel_color = render_trace(ray, false, id, max_working_threads);
            if (ray_planner.isNotEmpty(id)) {
                Printer().s("Stack with id ").i(id).s(" is not empty").nl();
            }

            canvas[i] = make_uchar3(pixel_color.x * 255, pixel_color.y * 255, pixel_color.z * 255);
        }
    }

    void render_cpu(uchar3* canvas) {
        int width = scene.camera.canvas_width;
        int height = scene.camera.canvas_height;

        cpm::vec3 origin = scene.camera.position;
        for (int j = 0; j < height; j++) {
            for (int i = 0; i < width; i++) {
                cpm::vec3 dir = scene.camera.generate_ray_direction(i, j);
                cpm::Ray ray(origin, dir);
                size_t id = i + j * width;
                cpm::vec3 pixel_color = render_trace(ray, false, 0);
                canvas[id] = make_uchar3(pixel_color.x * 255, pixel_color.y * 255, pixel_color.z * 255);
            }
            if (j % ((size_t)height / 50) == 0) {
                std::cout << "\tPixels filled: " << (j + 1) * width << " of " << width * height << std::endl;
            }
        }
        std::cout << "Rendering has ended" << std::endl;
    }
    __host__ __device__ void set_scene(Scene* scene) {
        this->scene = *scene;
    }

    __host__ __device__ void set_planner(RaytracePlanner* planner) {
        this->ray_planner = *planner;
    }

    __host__ __device__ void set_heap_pointer(PhotonMaxHeap* data) {
        heaps = data;
    }

    __host__ __device__ void set_photon_maps(PhotonGrid diffuse_grid, PhotonGrid specular_grid) {
        this->diffuse_grid = diffuse_grid;
        this->specular_grid = specular_grid;
    }
};