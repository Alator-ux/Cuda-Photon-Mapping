#include "GlobalParams.cuh"

/* Refractive index */
float GlobalParams::cpu_default_refractive_index;
__constant__ float GlobalParams::gpu_default_refractive_index;

__host__ void GlobalParams::set_default_refractive_index(float value) {
    cpu_default_refractive_index = value;
    cudaMemcpyToSymbol(gpu_default_refractive_index, &value, sizeof(float));
    checkCudaErrors(cudaGetLastError());
}

/* Max depth */
int GlobalParams::cpu_max_depth;
__constant__ int GlobalParams::gpu_max_depth;

__host__ void GlobalParams::set_max_depth(int value) {
    cpu_max_depth = value;
    cudaMemcpyToSymbol(gpu_max_depth, &value, sizeof(int));
    checkCudaErrors(cudaGetLastError());
}