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

/* Photon num */
int GlobalParams::cpu_global_photon_num;
__constant__ int GlobalParams::gpu_global_photon_num;

int GlobalParams::cpu_caustic_photon_num;
__constant__ int GlobalParams::gpu_caustic_photon_num;

__host__ void GlobalParams::set_photon_num(int global, int caustic) {
    cpu_global_photon_num = global;
    cudaMemcpyToSymbol(gpu_global_photon_num, &global, sizeof(int));
    checkCudaErrors(cudaGetLastError());

    cpu_caustic_photon_num = caustic;
    cudaMemcpyToSymbol(gpu_caustic_photon_num, &caustic, sizeof(int));
    checkCudaErrors(cudaGetLastError());
}

/* KdTree Chunk Size */
int GlobalParams::cpu_kd_tree_chunk_size              = 32;
__constant__ int GlobalParams::gpu_kd_tree_chunk_size = 32;

/* Predefined Device Params */
__constant__ uint GlobalParams::max_active_blocks;
__constant__ uint GlobalParams::max_active_threads;




__host__ void GlobalParams::set_medium_manager_parameters(MMInnerData* cpu_mm_inner_data_val, MMInnerData* gpu_mm_inner_data_val, idxtype mm_inner_capacity_val, 
                                                          MMInnerContainer* cpu_mm_outer_data_val, MMInnerContainer* gpu_mm_outer_data_val, idxtype mm_outer_capacity_val,
                                                          RayPlan* cpu_raytrace_planner_data_val, RayPlan* gpu_raytrace_planner_data_val, idxtype raytrace_planner_capacity_val) {

    set_medium_content_parameters(cpu_mm_inner_data_val, gpu_mm_inner_data_val, mm_inner_capacity_val);
    set_mm_inner_stack_parameters(cpu_mm_outer_data_val, gpu_mm_outer_data_val, mm_outer_capacity_val);
    set_rp_stack_parameters(cpu_raytrace_planner_data_val, gpu_raytrace_planner_data_val, raytrace_planner_capacity_val);
}