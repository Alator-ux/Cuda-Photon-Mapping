#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "vec3.cuh"
#include "CudaUtils.cuh"
#include "vec3.cuh"
#include "Defines.cuh"

namespace {
    constexpr float pi = 3.141592653589793;
}

struct Camera {
    cpm::vec3 position;
    cpm::vec3 forward; 
    cpm::vec3 right;   
    cpm::vec3 up;      

    float aspect_ratio;
    float vertical_fov;

    float viewport_width;   
    float viewport_height;  
    float focal_length;

    float yaw;
    float pitch;

    int canvas_width;
    int canvas_height;

    __host__ __device__ Camera() : position(0.f), forward(0.f), right(0.f), up(0.f), aspect_ratio(0.f), vertical_fov(0.f),
        viewport_height(0.f), viewport_width(0.f), focal_length(0.f), yaw(0.f), pitch(0.f), canvas_width(0), canvas_height(0) {}

    __host__ __device__ Camera(const Camera& other) : 
        position(other.position), forward(other.forward), right(other.right), up(other.up),
        aspect_ratio(other.aspect_ratio), vertical_fov(other.vertical_fov),
        viewport_width(other.viewport_width), viewport_height(other.viewport_height),
        focal_length(other.focal_length), yaw(other.yaw), pitch(other.pitch),
        canvas_width(other.canvas_width), canvas_height(other.canvas_height)
    { }

    __host__ __device__ Camera(
        const cpm::vec3& position, const cpm::vec3& direction, const cpm::vec3& up_vector,
        float vertical_fov_degrees, int canvas_width, int canvas_height
    ) : position(position), vertical_fov(vertical_fov_degrees),
        canvas_width(canvas_width), canvas_height(canvas_height)
    {
        aspect_ratio = (float)(canvas_width) / (float)(canvas_height);

        float theta = vertical_fov_degrees * pi / 180.0f;
        float h = tan(theta / 2);

        viewport_height = 2.0f * h;
        viewport_width = aspect_ratio * viewport_height;
        focal_length = 1.0f;

        cpm::vec3 normalized_direction = cpm::vec3::normalize(direction);
        forward = normalized_direction;
        right = cpm::vec3::normalize(cpm::vec3::cross(forward, up_vector));
        up = cpm::vec3::cross(right, forward);

        yaw = atan2(direction.z, direction.x) * 180.0f / pi;
        pitch = asin(direction.y) * 180.0f / pi;
    }

    __host__ __device__ cpm::vec3 generate_ray_direction(int pixel_x, int pixel_y) const { // TODO вычислить заранее для cuda
        float u = ((float)(pixel_x) + 0.5f) / canvas_width;
        float v = ((float)(pixel_y) + 0.5f) / canvas_height;
        
        /*cpm::vec3 horizontal = right * viewport_width;
        cpm::vec3 vertical = up * viewport_height;
        cpm::vec3 lower_left_corner = position + forward * focal_length - horizontal * 0.5f - vertical * 0.5f;*/
        cpm::vec3 horizontal = right * viewport_width;
        cpm::vec3 vertical = up * viewport_height;
        cpm::vec3 lower_left_corner = position.copy().add(forward).mult(focal_length)
            .sub(horizontal.copy().mult(0.5f)).sub(vertical.copy().mult(0.5f));

        // No need to normalize because of normalization in Ray constructor
        return lower_left_corner.add(horizontal.mult(u)).add(vertical.mult(v)).sub(position);
    }

    __host__ void move(const cpm::vec3& direction, float amount) {
        position += cpm::vec3::normalize(direction) * amount;
    }

    __host__ void rotate(float delta_yaw, float delta_pitch) {
        yaw += delta_yaw;
        pitch += delta_pitch;

        if (pitch > 89.0f) pitch = 89.0f;
        if (pitch < -89.0f) pitch = -89.0f;

        float yaw_radians = yaw * pi / 180.0f;
        float pitch_radians = pitch * pi / 180.0f;

        cpm::vec3 new_forward;
        new_forward[0] = cos(yaw_radians) * cos(pitch_radians);
        new_forward[1] = sin(pitch_radians);
        new_forward[2] = sin(yaw_radians) * cos(pitch_radians);

        forward = cpm::vec3::normalize(new_forward);

        // Пересчёт right и up
        right = cpm::vec3::normalize(cpm::vec3::cross(forward, cpm::vec3(0, 1, 0)));
        up = cpm::vec3::cross(right, forward);
    }

    __host__ void to_device(Camera** ptr) {
        if ((*ptr) == nullptr) {
            cudaMalloc(ptr, sizeof(Camera));
        }

        checkCudaErrors(cudaMemcpy((*ptr), this, sizeof(Camera), cudaMemcpyHostToDevice));
    }
};
