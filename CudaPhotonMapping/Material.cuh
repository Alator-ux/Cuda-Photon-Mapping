#pragma once
#include <crt/host_defines.h>
#include "vec3.cuh"
struct Material {
    cpm::vec3 ambient = cpm::vec3(0.2f);
    cpm::vec3 diffuse = cpm::vec3(1.f);
    cpm::vec3 specular = cpm::vec3(0.f);
    cpm::vec3 emission = cpm::vec3(0.0);
    // Optical density, also known as index of refraction
    float refr_index = 1.f;
    float opaque = 1.f;
    float shininess = 1.f;
    __host__ __device__ Material(const Material& other) {
        this->ambient = other.ambient;
        this->diffuse = other.diffuse;
        this->specular = other.specular;
        this->emission = other.emission;
        this->opaque = other.opaque;
        this->shininess = other.shininess;
    }
    __host__ __device__ Material(cpm::vec3 ambient = cpm::vec3(0.2f),
        cpm::vec3 diffuse = cpm::vec3(1.0f), cpm::vec3 specular = cpm::vec3(1.0f),
        cpm::vec3 emission = cpm::vec3(0.0f), float shininess = 16.0f,
        float roughless = 0.3) {
        this->ambient = ambient;
        this->diffuse = diffuse;
        this->specular = specular;
        this->emission = emission;
        this->shininess = shininess;
    }
};