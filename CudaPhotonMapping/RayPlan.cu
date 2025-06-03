#include "RayPlan.cuh"
#include "Defines.cuh"

RayPlan* cpu_raytrace_planner_data;
__constant__ RayPlan* gpu_raytrace_planner_data;

idxtype cpu_raytrace_planner_capacity;
__constant__ idxtype gpu_raytrace_planner_capacity;

__host__ void set_rp_stack_parameters(RayPlan* cpu_raytrace_planner_data_val, RayPlan* gpu_raytrace_planner_data_val, idxtype raytrace_planner_capacity_val) {
    cpu_raytrace_planner_data = cpu_raytrace_planner_data_val;
    cpu_raytrace_planner_capacity = raytrace_planner_capacity_val;

    cudaMemcpyToSymbol(gpu_raytrace_planner_data, &gpu_raytrace_planner_data_val, sizeof(RayPlan*));
    checkCudaErrors(cudaGetLastError());
    cudaMemcpyToSymbol(gpu_raytrace_planner_capacity, &raytrace_planner_capacity_val, sizeof(idxtype));
    checkCudaErrors(cudaGetLastError());
}