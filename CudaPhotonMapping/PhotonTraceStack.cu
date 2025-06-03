#include "PhotonTraceStack.cuh""

MMInnerData* cpu_pt_stack_data;
__constant__ MMInnerData* gpu_pt_stack_data;

idxtype cpu_pt_stack_capacity;
__constant__ idxtype gpu_pt_stack_capacity;

__host__ void set_pt_stack_parameters(MMInnerData* cpu_pt_stack_data_val,
    MMInnerData* gpu_pt_stack_data_val, idxtype pt_stack_capacity_val) {
    cpu_pt_stack_data = cpu_pt_stack_data_val;
    cpu_pt_stack_capacity = pt_stack_capacity_val;

    cudaMemcpyToSymbol(gpu_pt_stack_data, &gpu_pt_stack_data_val, sizeof(MMInnerData*));
    checkCudaErrors(cudaGetLastError());
    cudaMemcpyToSymbol(gpu_pt_stack_capacity, &pt_stack_capacity_val, sizeof(idxtype));
    checkCudaErrors(cudaGetLastError());
}