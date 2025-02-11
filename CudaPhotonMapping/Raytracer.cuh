#pragma once
#include "CudaUtils.cuh"
#include "Scene.cuh"
#include "MediumManager.cuh"
#include "DeepLookStack.cuh"
#include "RaytracePlanner.cuh"

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

//    __device__ float* mediums_stack;
    __host__ __device__ cpm::vec3 render_trace(cpm::Ray current_ray, bool in_object, size_t stack_id) {
        int new_depth = 0;
        int current_depth = -1;
        cpm::vec3 accumulated_color(0.f);
        cpm::vec3 new_ray_coef(1.f);
        if (stack_id == 2272) {
            printf("a");
        }
        bool replace_medium = false;
        while ((current_depth < GlobalParams::max_depth() && current_depth != new_depth)
                || ray_planner.pop_refraction(stack_id, replace_medium, current_ray, new_depth, new_ray_coef, in_object)) {

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
                int important_ls = 0;
                cpm::vec3 tnormal, tinter_p;
                Model* timodel;
                cpm::Ray tray;

                int ls_number = scene.light_sources_number;
                LightSource* light_sources = scene.light_sources;
                for (int i = 0; i < ls_number; i++) {
                    const LightSource* ls = light_sources + i;
                    tray.origin = ls->position;
                    tray.direction = cpm::vec3::normalize(inter_p - ls->position); // point <- ls
                    tray.origin += tray.direction * 0.001f;
                    if (find_intersection(tray, in_object, timodel, tnormal, tinter_p) && inter_p.equal(tinter_p)) {
                        important_ls++;
                        re += max(cpm::vec3::dot(normal, -tray.direction), 0.f);
                    }
                }

                re /= important_ls == 0 ? 1 : important_ls;
                auto di_op = mat.diffuse * mat.opaque;

                re *= di_op;
                current_color += re;
            }

            current_color += mat.emission;

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
                    ray_planner.push_refraction(stack_id, replace_medium,
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
        
        accumulated_color.clamp_min(1.f);
        return accumulated_color;
    }

   public:
  
    __device__ void render_gpu(uchar3* canvas, int width, int height) {
        /*int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;
        int local_id = threadIdx.y * blockDim.x + threadIdx.x;
        if (x >= width || y >= height) return;*/

        int id = blockIdx.x * blockDim.x + threadIdx.x;
        if (id >= width * height) {
            return;
        }
        int x = id % width;
        int y = id / width;
        
        int local_id = threadIdx.x;

        cpm::Ray ray(scene.camera.position, scene.camera.generate_ray_direction(x, y));

        /*Model* m;
        cpm::vec3 t1, t2;
        for (int i = 0; i < 10; i++) {
            ray.origin -= cpm::vec3(0, 0, 0.001f);
            bool res = find_intersection(ray, false, m, t1, t2);
            if (res) {
                canvas[id] = make_uchar3((255 * i) % 255, t1.x * 255, t2.y * 255);
            }
            else {
                canvas[id] = make_uchar3(0, 0, 0);
            }
        }*/

        /*float outf;
        cpm::vec3 outv;
        for (int i = 0; i < 2188; i++) {
            bool res = traingle_intersection(ray, p1, p2, p1, outf, outv);
            if (res) {
                canvas[id] = make_uchar3(255, 255 ,255);
            }
            else {
                canvas[id] = make_uchar3(0, 0, 0);
            }
        }*/

        //find_intersection(ray, false, intersection_infos + local_id
        cpm::vec3 pixel_color = render_trace(ray, false, id);
        if (ray_planner.isNotEmpty(id)) {
            Printer().s("Stack with id ").i(id).s(" is not empty").nl();
        }
        canvas[id] = make_uchar3(pixel_color.x * 255, pixel_color.y * 255, pixel_color.z * 255);
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
                cpm::vec3 pixel_color = render_trace(ray, false, id);
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

    __host__ void initialize_cpu(size_t pixels_number) {
        constexpr int max_depth = 10;

        ray_planner.intialize_cpu(max_depth, pixels_number);
    }
};