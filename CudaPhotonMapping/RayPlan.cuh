#pragma once
#include "Ray.cuh"
#include "Defines.cuh"

struct RayPlan {
	cpm::Ray ray;
	int depth;
	cpm::vec3 coef;
	bool in_object;
};

/* Planner Stack */
extern RayPlan* cpu_raytrace_planner_data;
extern __constant__ RayPlan* gpu_raytrace_planner_data;

__host__ __device__ __forceinline__ RayPlan* raytrace_planner_data() {
#ifdef __CUDA_ARCH__
	return gpu_raytrace_planner_data;
#else
	return cpu_raytrace_planner_data;
#endif
}

extern idxtype cpu_raytrace_planner_capacity;
extern __constant__ idxtype gpu_raytrace_planner_capacity;

__host__ __device__ __forceinline__ idxtype raytrace_planner_capacity() {
#ifdef __CUDA_ARCH__
	return gpu_raytrace_planner_capacity;
#else
	return cpu_raytrace_planner_capacity;
#endif
}

__host__ void set_rp_stack_parameters(RayPlan* cpu_raytrace_planner_data_val, RayPlan* gpu_raytrace_planner_data_val, idxtype raytrace_planner_capacity_val);