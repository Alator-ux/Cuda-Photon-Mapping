#pragma once
#include "CudaUtils.cuh"
#include "Scene.cuh"
#include "MediumManager.cuh"

namespace {
    using namespace std;
}

class Raytracer {
    Scene scene;
//    MediumManager* medium_manager;

    __host__ __device__ bool find_intersection(const cpm::Ray& ray, bool reverse_normal, 
        Model*& out_incident_model, cpm::vec3& out_normal, cpm::vec3& out_intersection_point) {
        float intersection = 0.f;
        out_incident_model = nullptr;
        size_t ii0, ii1, ii2;

        int models_number = scene.models_number;
        Model* models = scene.models;
        for (int i = 0; i < models_number; i++) {
            Model* model = models + i;

            float temp_inter;
            size_t tii0, tii1, tii2;
            cpm::vec3 tnormal;
            bool succ = model->intersection(ray, false, temp_inter,
                tii0, tii1, tii2, tnormal);
            if (succ && (intersection == 0.f || temp_inter < intersection)) {
                intersection = temp_inter;
                ii0 = tii0;
                ii1 = tii1;
                ii2 = tii2;
                out_incident_model = model;
                out_normal = tnormal;
            }
        }
        if (out_incident_model == nullptr) {
            return false;
        }
        out_intersection_point = ray.origin + ray.direction * intersection;
        if (reverse_normal && cpm::vec3::dot(ray.direction, out_normal) > 0) {
            out_normal *= -1.f;
        }
        return true;
    }

    __host__ __device__ cpm::vec3 render_trace(const cpm::Ray& ray, bool in_object, int depth) {
        cpm::vec3 res(0.f);

        const int max_depth = 1;

        cpm::Ray current_ray = ray;
        cpm::vec3 accumulated_color(0.f);
        cpm::vec3 coef_after_reflection(1.f);
        for (int current_depth = 0; current_depth < max_depth; current_depth++) {
            cpm::vec3 normal, inter_p;
            Model* imodel;

            if (!find_intersection(current_ray, in_object, imodel, normal, inter_p)) {
                break;
            }

            Material mat = *imodel->get_material();

            if (!mat.diffuse.is_zero()) {
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

                accumulated_color += re * di_op * coef_after_reflection;
            }

            accumulated_color += mat.emission * coef_after_reflection;

            if (!in_object && !mat.specular.is_zero()) {
                cpm::Ray nray = current_ray.reflect(inter_p, normal);
                float coef = pow(max(cpm::vec3::dot(nray.direction, -current_ray.direction), 0.0f), mat.shininess);
             
                coef_after_reflection *= mat.specular * mat.opaque * coef;
                current_ray = nray;
            }
        }

        res = accumulated_color;
        return res;
    }

   public:
    __device__ void render_gpu(uchar3* canvas, int width, int height) {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;
        if (x >= width || y >= height) return;

        cpm::vec3 origin = scene.camera.position;

        cpm::vec3 dir = scene.camera.generate_ray_direction(x, y);
        cpm::Ray ray(origin, dir);
        //cpm::vec3 pixel_color = render_trace(ray, false, 0);
        cpm::vec3 t1, t2;
        Model* m1;
        find_intersection(ray, false, m1, t1, t2);
        //canvas[x + y * width] = make_uchar3(pixel_color.x() * 255, pixel_color.y() * 255, pixel_color.z() * 255);
    }

    void render_cpu(uchar3* canvas) {
        int width = scene.camera.canvas_width;
        int height = scene.camera.canvas_height;

        cpm::vec3 origin = scene.camera.position;
        for (int j = 0; j < height; j++) {
            for (int i = 0; i < width; i++) {
                cpm::vec3 dir = scene.camera.generate_ray_direction(i, j);
                cpm::Ray ray(origin, dir);
                cpm::vec3 pixel_color = render_trace(ray, false, 0);
                canvas[i + j * width] = make_uchar3(pixel_color.x() * 255, pixel_color.y() * 255, pixel_color.z() * 255);
            }
            if (j % ((size_t)height / 50) == 0) {
                std::cout << "\tPixels filled: " << (j + 1) * width << " of " << width * height << std::endl;
            }
        }
        std::cout << "Rendering has ended" << std::endl;
    }
    void set_scene(Scene* scene) {
        this->scene = *scene;
    }
};