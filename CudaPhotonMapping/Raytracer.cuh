#pragma once
#include "CudaUtils.cuh"
#include "Scene.cuh"
#include "MediumManager.cuh"

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

    __host__ __device__ bool find_intersection(const cpm::Ray& ray, bool reverse_normal, IntersectionInfo* intersection_info) {
        float intersection = 0.f;
        Model* out_incident_model = nullptr;
        cpm::vec3 out_normal, out_intersection_point;
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

        intersection_info->incident_model = out_incident_model;
        intersection_info->intersection_point = out_intersection_point;
        intersection_info->normal = out_normal;

        return true;
    }

    __host__ __device__ cpm::vec3 render_trace(const cpm::Ray& ray, bool in_object, int depth) {
        cpm::vec3 res(0.f);

        const int max_depth = 1;

        cpm::Ray current_ray = ray;
        cpm::vec3 accumulated_color(0.f);
        cpm::vec3 coef_after_reflection(1.f);
        //for (int current_depth = 0; current_depth < max_depth; current_depth++) {
        //    cpm::vec3 normal, inter_p;
        //    Model* imodel;

        //    if (!find_intersection(current_ray, in_object, imodel, normal, inter_p)) {
        //        break;
        //    }

        //    Material mat = *imodel->get_material();

        //    if (!mat.diffuse.is_zero()) {
        //        cpm::vec3 re(0.f);
        //        int important_ls = 0;
        //        cpm::vec3 tnormal, tinter_p;
        //        Model* timodel;
        //        cpm::Ray tray;

        //        int ls_number = scene.light_sources_number;
        //        LightSource* light_sources = scene.light_sources;
        //        for (int i = 0; i < ls_number; i++) {
        //            const LightSource* ls = light_sources + i;
        //            tray.origin = ls->position;
        //            tray.direction = cpm::vec3::normalize(inter_p - ls->position); // point <- ls
        //            tray.origin += tray.direction * 0.001f; 
        //            if (find_intersection(tray, in_object, timodel, tnormal, tinter_p) && inter_p.equal(tinter_p)) {
        //                important_ls++;
        //                re += max(cpm::vec3::dot(normal, -tray.direction), 0.f);
        //            }
        //        }

        //        re /= important_ls == 0 ? 1 : important_ls;
        //        auto di_op = mat.diffuse * mat.opaque;

        //        accumulated_color += re * di_op * coef_after_reflection;
        //    }

        //    accumulated_color += mat.emission * coef_after_reflection;

        //    if (!in_object && !mat.specular.is_zero()) {
        //        cpm::Ray nray = current_ray.reflect(inter_p, normal);
        //        float coef = pow(max(cpm::vec3::dot(nray.direction, -current_ray.direction), 0.0f), mat.shininess);
        //     
        //        coef_after_reflection *= mat.specular * mat.opaque * coef;
        //        current_ray = nray;
        //    }
        //}

        res = accumulated_color;
        return res;
    }








    /* floats operation:
    *  86 + some index operations!!!!
    *  86 * 2188 * (100*100) = 1.882 לכנה
    */

    __host__ __device__ bool traingle_intersection(const cpm::Ray& ray,
            const cpm::vec3& v0, const cpm::vec3& v1, const cpm::vec3& v2,
            float& out_ray_parameter, cpm::vec3& out_uvw) const {
        out_ray_parameter = 0.f;
        // compute the plane's normal
        cpm::vec3 v0v1 = v1 - v0; //! 3 floats
        cpm::vec3 v0v2 = v2 - v0; //! 3 floats
        // no need to normalize
        // Normal of the triangle plane
        cpm::vec3 N = cpm::vec3::cross(v0v1, v0v2); //! 9 floats
        float denom = cpm::vec3::dot(N, N); //! 5 floats

        // Step 1: finding P

        // check if the ray and plane are parallel.
        float NdotRayDirection = cpm::vec3::dot(N, ray.direction); //! 5 floats
        if (fabsf(NdotRayDirection) < model_eps) // almost 0
            return false; // they are parallel so they don't intersect! 

        // compute t (equation 3)
        out_ray_parameter = (cpm::vec3::dot(N, v0) - cpm::vec3::dot(N, ray.origin)) / NdotRayDirection;//! 5+5+1=11 floats
         //check if the triangle is behind the ray
        if (out_ray_parameter < 0) return false; // the triangle is behind

        // compute the intersection point using equation 1
        cpm::vec3 P = ray.origin.copy().add(out_ray_parameter).mult(ray.direction); //! 3+3=6 floats

        // Step 2: inside-outside test

        // edge 0
        cpm::vec3 edge0 = v1 - v0; //! 3 floats
        cpm::vec3 vp0 = P - v0; //! 3 floats
        // vector perpendicular to triangle's plane
        float w = cpm::vec3::dot(N, edge0.cross(vp0)); //! 9+5=14 floats
        if (w < 0) return false; // P is on the right side

        // edge 2
        cpm::vec3 edge2 = v0 - v2; //! 3 floats
        cpm::vec3 vp2 = P - v2; //! 3 floats
        float v = cpm::vec3::dot(N, edge2.cross(vp2)); //! 9+5=14 floats
        if (v < 0) return false; // P is on the right side;
        w /= denom; //! 1 floats
        v /= denom; //! 1 floats
        float u = 1.f - v - w; //! 2 floats
        if (u < 0) {
            return false;
        }
        out_uvw[0] = u;
        out_uvw[1] = v;
        out_uvw[2] = w;
        return true; // this ray hits the triangle
    }





   public:
  
    __device__ void render_gpu(uchar3* canvas, int width, int height) {
        extern __shared__ int intersection_infos[];
        /*int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;
        int local_id = threadIdx.y * blockDim.x + threadIdx.x;
        if (x >= width || y >= height) return;*/

        int id = blockIdx.x * blockDim.x + threadIdx.x;
        if (id >= width * height) {
            atomicAdd(&intersection_infos[0], 1);
            printf("total %i", intersection_infos[0]);
            return;
        }
        int x = id % width;
        int y = id / width;
        
        int local_id = threadIdx.x;

        cpm::vec3 origin = scene.camera.position;

        cpm::vec3 dir = scene.camera.generate_ray_direction(x, y);
        cpm::Ray ray(origin, dir);

        cpm::vec3 p1 = scene.models->mci.positions[0];
        cpm::vec3 p2 = scene.models->mci.positions[0];
        /*Model* m;
        cpm::vec3 t1, t2;
        find_intersection(ray, false, m, t1,t2);*/
        float outf;
        cpm::vec3 outv;
        for (int i = 0; i < 2188; i++) {
            bool res = traingle_intersection(ray, p1, p2, p1, outf, outv);
            if (res) {
                canvas[id] = make_uchar3(255, 255 ,255);
            }
            else {
                canvas[id] = make_uchar3(0, 0, 0);
            }
        }

        //find_intersection(ray, false, intersection_infos + local_id);
        /*cpm::vec3 pixel_color = render_trace(ray, false, 0);
        canvas[id] = make_uchar3(pixel_color.x() * 255, pixel_color.y() * 255, pixel_color.z() * 255);*/
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