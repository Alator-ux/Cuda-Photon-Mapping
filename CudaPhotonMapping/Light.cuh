#pragma once
#include <crt/host_defines.h>
#include "vec3.cuh"

struct Model;

struct LightSource {
    cpm::vec3 intensity = cpm::vec3(1.f);
    cpm::vec3 position;
    cpm::vec3 ambient;
    cpm::vec3 diffuse;
    cpm::vec3 specular;
    Model* model_owner = nullptr;
    __host__ __device__ LightSource(cpm::vec3 position = cpm::vec3(0.f), cpm::vec3 ambient = cpm::vec3(1.0),
        cpm::vec3 diffuse = cpm::vec3(1.0), cpm::vec3 specular = cpm::vec3(1.0)) {
        this->position = position;
        this->ambient = ambient;
        this->diffuse = diffuse;
        this->specular = specular;
    }
};

struct DirectionLight : public LightSource {
    cpm::vec3 direction;
    __host__ __device__ DirectionLight(cpm::vec3 position = cpm::vec3(0.f), cpm::vec3 direction = cpm::vec3(0.0), cpm::vec3 ambient = cpm::vec3(1.0),
        cpm::vec3 diffuse = cpm::vec3(1.0), cpm::vec3 specular = cpm::vec3(1.0)) {
        this->direction = direction;
        this->ambient = ambient;
        this->diffuse = diffuse;
        this->specular = specular;
    }
};