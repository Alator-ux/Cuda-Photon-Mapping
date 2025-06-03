#pragma once
#include <crt/host_defines.h>
#include "Defines.cuh"
#include "RayPlan.cuh"
#include "MediumContent.cuh"
#include "MMInnerStack.cuh"


namespace GlobalParams {
    /* ===== Refractive index ===== */
    extern float cpu_default_refractive_index;
    extern __constant__ float gpu_default_refractive_index;

    __host__ __device__ __forceinline__ float inline default_refractive_index() {
#ifdef __CUDA_ARCH__
        return gpu_default_refractive_index;
#else
        return cpu_default_refractive_index;
#endif  __CUDA_ARCH__
    }

    __host__ void set_default_refractive_index(float value);

    /* ===== Max depth ===== */
    extern int cpu_max_depth;
    extern __constant__ int gpu_max_depth;

    __host__ __device__ __forceinline__ int inline max_depth() {
#ifdef __CUDA_ARCH__
        return gpu_max_depth;
#else
        return cpu_max_depth;
#endif
    }

    __host__ void set_max_depth(int value);

    /* ===== Photon num ===== */
    extern int cpu_global_photon_num;
    extern __constant__ int gpu_global_photon_num;

    __host__ __device__ __forceinline__ int inline global_photon_num() {
#ifdef __CUDA_ARCH__
        return gpu_global_photon_num;
#else
        return cpu_global_photon_num;
#endif
    }

    extern int cpu_caustic_photon_num;
    extern __constant__ int gpu_caustic_photon_num;

    __host__ __device__ __forceinline__ int inline caustic_photon_num() {
#ifdef __CUDA_ARCH__
        return gpu_caustic_photon_num;
#else
        return cpu_caustic_photon_num;
#endif
    }

    __host__ void set_photon_num(int global, int caustic);

    /* ===== KdTree Chunk Size ===== */
    extern int cpu_kd_tree_chunk_size;
    extern __constant__ int gpu_kd_tree_chunk_size;
    __host__ __device__ __forceinline__ int inline kd_tree_chunk_size() {
#ifdef __CUDA_ARCH__
        return gpu_kd_tree_chunk_size;
#else
        return cpu_kd_tree_chunk_size;
#endif
    }

    /* ===== PREDEFINED ===== */
    extern __constant__ uint max_active_blocks;
    extern __constant__ uint max_active_threads;

    /* ===== Raytrace Planner, Medium Manager Inner And Outer Content ===== */


//
//#ifdef __CUDA_ARCH__
//#define MMInnerContainerData GlobalParams::gpu_mm_inner_data
//#define MMInnerContainerCapacity GlobalParams::gpu_mm_inner_capacity
//#else
//#define MMInnerContainerData GlobalParams::cpu_mm_inner_data
//#define MMInnerContainerCapacity GlobalParams::cpu_mm_inner_capacity
//#endif
//
//#ifdef __CUDA_ARCH__
//#define MMOuterContainerData GlobalParams::gpu_mm_outer_data
//#define MMOuterContainerCapacity GlobalParams::gpu_mm_outer_capacity
//#else
//#define MMOuterContainerData GlobalParams::cpu_mm_outer_data
//#define MMOuterContainerCapacity GlobalParams::cpu_mm_outer_capacity
//#endif


    __host__ void set_medium_manager_parameters(MMInnerData* cpu_mm_inner_data_val, MMInnerData* gpu_mm_inner_data_val, idxtype mm_inner_capacity_val,
                                                MMInnerContainer* cpu_mm_outer_data_val, MMInnerContainer* gpu_mm_outer_data_val, idxtype mm_outer_capacity,
                                                RayPlan* cpu_raytrace_planner_data_val, RayPlan* gpu_raytrace_planner_data_val, idxtype raytrace_planner_capacity_val);
}