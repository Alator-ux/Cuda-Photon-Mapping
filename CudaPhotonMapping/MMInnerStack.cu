#include "MmInnerStack.cuh"

MMInnerContainer* cpu_mm_outer_data;
__constant__ MMInnerContainer* gpu_mm_outer_data;


idxtype cpu_mm_outer_capacity;
__constant__ idxtype gpu_mm_outer_capacity;

__host__ void set_mm_inner_stack_parameters(MMInnerContainer* cpu_mm_outer_data_val, MMInnerContainer* gpu_mm_outer_data_val, idxtype mm_outer_capacity_val) {
    cpu_mm_outer_data = cpu_mm_outer_data_val;
    cpu_mm_outer_capacity = mm_outer_capacity_val;

    cudaMemcpyToSymbol(gpu_mm_outer_data, &gpu_mm_outer_data_val, sizeof(MMInnerContainer*));
    checkCudaErrors(cudaGetLastError());
    cudaMemcpyToSymbol(gpu_mm_outer_capacity, &mm_outer_capacity_val, sizeof(idxtype));
    checkCudaErrors(cudaGetLastError());
}