#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "Model.cuh"
#include "Camera.cuh"

struct Scene {
    Model* models;
    int models_number;
    LightSource* light_sources;
    int light_sources_number;
    Camera camera;
    __host__ __device__ Scene() : models(nullptr), models_number(0), light_sources(nullptr), light_sources_number(0), camera() {}
    __host__ __device__ Scene(Model* models, int models_number, LightSource* light_sources, int light_sources_number, Camera camera) 
        : models(models), models_number(models_number),
        light_sources(light_sources), light_sources_number(light_sources_number),
        camera(camera) {}
};