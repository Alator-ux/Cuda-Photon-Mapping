#include "MediumContent.cuh"

MMInnerData* cpu_mm_inner_data;
__constant__ MMInnerData* gpu_mm_inner_data;

idxtype cpu_mm_inner_capacity;
__constant__ idxtype gpu_mm_inner_capacity;

__host__ void set_medium_content_parameters(MMInnerData* cpu_mm_inner_data_val, MMInnerData* gpu_mm_inner_data_val, idxtype mm_inner_capacity_val) {
    cpu_mm_inner_data = cpu_mm_inner_data_val;
    cpu_mm_inner_capacity = mm_inner_capacity_val;

    cudaMemcpyToSymbol(gpu_mm_inner_data, &gpu_mm_inner_data_val, sizeof(MMInnerData*));
    checkCudaErrors(cudaGetLastError());
    cudaMemcpyToSymbol(gpu_mm_inner_capacity, &mm_inner_capacity_val, sizeof(idxtype));
    checkCudaErrors(cudaGetLastError());
}