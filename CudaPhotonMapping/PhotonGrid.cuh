#pragma once
#include "Photon.cuh"
#include "Defines.cuh"
#include "AABB.cuh"
#include "PhotonMaxHeap.cuh"

struct GridSize3D {
    int x, y, z;
};


class PhotonGrid {
    cpm::vec3* p_positions;
    cpm::vec3* p_directions;
    cpm::vec3* p_powers;
    int num_photons;       
    AABB scene_aabb;
    GridSize3D grid_size;  
    float cell_size;       
    uint* cell_starts;   
    uint* cell_sizes;    

    void compute_grid_and_cell_size(int desired_photons_per_cell);

    void analyze_density(int desired_photons_per_cell, cpm::Photon* photons);

    void build_hash_grid(uint max_photons_per_cell, cpm::Photon* photons);

    // haha
    __host__ __device__ void find_nearests_in_cell(uint cell_idx, const cpm::vec3& location_point, float radius, int num_to_find,
        PhotonMaxHeap& maxheap, uint array_idx, uint array_cap);

public:
    __host__ __device__ PhotonGrid() : p_positions(nullptr), p_directions(nullptr), p_powers(nullptr), num_photons(0),
        scene_aabb(),
        grid_size({ 0, 0, 0 }), cell_size(0.0f),
        cell_starts(nullptr), cell_sizes(nullptr) {}

    __host__ __device__ PhotonGrid(const PhotonGrid& other) : p_positions(other.p_positions), p_directions(other.p_directions), p_powers(other.p_powers),
        num_photons(other.num_photons),
        scene_aabb(other.scene_aabb), grid_size(other.grid_size), cell_size(other.cell_size),
        cell_starts(other.cell_starts), cell_sizes(other.cell_sizes) {}

    __host__ __device__ PhotonGrid(int num_photons, AABB scene_aabb)
        : p_positions(nullptr), p_directions(nullptr), p_powers(nullptr), num_photons(num_photons),
        scene_aabb(scene_aabb),
        grid_size({ 0, 0, 0 }), cell_size(0.0f),
        cell_starts(nullptr), cell_sizes(nullptr) {}


    void build(cpm::Photon* photons, int desired_photons_per_cell = 1000, uint max_photons_per_cell = 5000) {
        printf("\n-------------\nPhoton map with %i photons construction started\n", num_photons);
        if (num_photons == 0 || !photons) {
            printf("Error: No photons to build grid.\n");
            return;
        }

        analyze_density(desired_photons_per_cell, photons);

        cudaMalloc(&p_positions, num_photons * sizeof(cpm::vec3));
        cudaMalloc(&p_directions, num_photons * sizeof(cpm::vec3));
        cudaMalloc(&p_powers, num_photons * sizeof(cpm::vec3));

        build_hash_grid(max_photons_per_cell, photons);
        printf("Photon map construction ended\n-------------\n", num_photons);
    }

    __host__ __device__ bool find_nearests(const cpm::vec3& location_point, float radius, int num_to_find,
        PhotonMaxHeap& maxheap, uint array_idx, uint array_cap);

    __host__ __device__ cpm::vec3* get_photon_positions() { return p_positions; }
    __host__ __device__ cpm::vec3* get_photon_directions() { return p_directions; }
    __host__ __device__ cpm::vec3* get_photon_powers() { return p_powers; }

    __host__ PhotonGrid copy_to_cpu() {
        PhotonGrid result;
        result.num_photons = num_photons;
        result.scene_aabb = scene_aabb;     
        result.grid_size = grid_size;       
        result.cell_size = cell_size;

        uint photon_bytes = sizeof(cpm::vec3) * num_photons;
        result.p_positions = (cpm::vec3*)malloc(photon_bytes);
        result.p_directions = (cpm::vec3*)malloc(photon_bytes);
        result.p_powers = (cpm::vec3*)malloc(photon_bytes);
        cudaMemcpy(result.p_positions, p_positions, photon_bytes, cudaMemcpyDeviceToHost);
        cudaMemcpy(result.p_directions, p_directions, photon_bytes, cudaMemcpyDeviceToHost);
        cudaMemcpy(result.p_powers, p_powers, photon_bytes, cudaMemcpyDeviceToHost);

        int cell_count = grid_size.x * grid_size.y * grid_size.z;
        result.cell_starts = (uint*)malloc(sizeof(uint) * cell_count);
        result.cell_sizes = (uint*)malloc(sizeof(uint) * cell_count);
        cudaMemcpy(result.cell_starts, cell_starts, sizeof(uint) * cell_count, cudaMemcpyDeviceToHost);
        cudaMemcpy(result.cell_sizes, cell_sizes, sizeof(uint) * cell_count, cudaMemcpyDeviceToHost);
        checkCudaErrors(cudaGetLastError());

        return result;
    }
};