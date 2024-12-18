#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <string>
#include "vec3.cuh"
#include "vec2.cuh"
#include "Material.cuh"
#include <vector>

namespace cpm {
    template<typename T>
    __host__ __device__ void swap(T& first, T& second) {
        T temp = second;
        second = first;
        first = temp;
    }
};

enum ModelType {
    Triangle, Quad, Polygon
};

int model_type_to_primitive_size(ModelType type);

// Нужна для конструкции класса модели
struct ModelConstructInfo {
private:
    __host__ __device__ __forceinline__ void copy_values(const ModelConstructInfo& other) {
        this->size = other.size;
        this->primitives_size = other.primitives_size;
        this->type = other.type;
        this->material = other.material;
        this->smooth = other.smooth;
        this->positions = other.positions;
        this->texcoords = other.texcoords;
        this->normals = other.normals;
    }
public:
    cpm::vec3* positions;
    cpm::vec2* texcoords;
    cpm::vec3* normals;
    int size;
    int primitives_size;
    Material material;
    ModelType type;
    bool smooth = false;
    __host__ __device__ ModelConstructInfo() : size(0), primitives_size(0), material(), type(),
        positions(nullptr), texcoords(nullptr), normals(nullptr) {}
    __host__ __device__ ModelConstructInfo(const ModelConstructInfo& other) {
        copy_values(other);
    }
    __host__ __device__ ModelConstructInfo(ModelConstructInfo&& other) noexcept {
        this->swap(other);
    }
    __host__ __device__ void swap(ModelConstructInfo& other) {
        cpm::swap(this->size, other.size);
        cpm::swap(this->primitives_size, other.primitives_size);
        cpm::swap(this->material, other.material);
        cpm::swap(this->type, other.type);
        cpm::swap(this->smooth, other.smooth);
        cpm::swap(this->positions, other.positions);
        cpm::swap(this->texcoords, other.texcoords);
        cpm::swap(this->normals, other.normals);
    }
    __host__ __device__ ModelConstructInfo& operator=(const ModelConstructInfo& other) {
        copy_values(other);
        return *this;
    }
    __host__ __device__ ModelConstructInfo& operator=(ModelConstructInfo&& other) noexcept {
        this->swap(other);
        return *this;
    }
    static ModelConstructInfo* vector_to_device(const std::vector<ModelConstructInfo>& mcis);
};