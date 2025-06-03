#pragma once
#include <crt/host_defines.h>
#include <cuda_runtime.h>
#include "MediumManager.cuh"
#include "Ray.cuh"
#include "RayPlan.cuh"
#include "RayPlanStack.cuh"

class RaytracePlanner {
private:
	uint array_size;
	MediumManager* medium_managers;
	cpm::rayplan_stack* planner;
public:
	__host__ __device__ void initialize(cpm::rayplan_stack* planner, MediumManager* medium_managers, uint array_size);
	static __host__ cpm::pair<RaytracePlanner*, RaytracePlanner*> initialize(uint array_size, int max_depth, int max_medium_depth, float default_refr_index);

	__host__ __device__ RaytracePlanner& operator=(const RaytracePlanner& other)
	{
		if (this == &other)
			return *this;

		this->medium_managers = other.medium_managers;
		this->planner = other.planner;
		this->array_size = other.array_size;

		return *this;
	}

	__host__ __device__ cpm::Tuple3<float, float, bool> get_refractive_indices(uint stack_id, int model_id, float model_refractive_index) {
		return medium_managers[stack_id].get_refractive_indices(stack_id, array_size, model_id, model_refractive_index);
	}
	__host__ __device__ void push_refraction(uint stack_id, bool& max_level,
		const cpm::Ray& ray, int depth, const cpm::vec3& coef,
		int model_id, float model_refractive_index, float model_opaque, bool in_object) {
		if (stack_id >= array_size) {
			printf("Stack id (%u) >= array size (%u) in RaytracePlanner, function push_refraction", stack_id, array_size);
			return;
		}
		/*if (stack_id == 243251) {
			printf("c");
		}*/
		medium_managers[stack_id].increase_depth(stack_id, array_size, max_level);
		medium_managers[stack_id].inform(stack_id, array_size, model_id, model_refractive_index, max_level);

		planner[stack_id].push(stack_id, array_size, {ray, depth + 1, coef * (1.f - model_opaque), in_object});

	}
	__host__ __device__ bool pop_refraction(uint stack_id, bool& max_level, 
		cpm::Ray& ray, int& depth, cpm::vec3& coef, bool& in_object) {
		// no refractions by this ray
		//if (stack_id == 243251) {
		//	printf("c");
		//}
		medium_managers[stack_id].reduce_depth(stack_id, array_size, max_level);
		if (stack_id >= array_size) {
			printf("Stack id (%u) >= array size (%u) in RaytracePlanner, function pop_refraction", stack_id, array_size);
			return false;
		}
		if (planner[stack_id].isEmpty()) {
			return false;
		}
		RayPlan plan = planner[stack_id].pop(stack_id, array_size);
		ray = plan.ray;
		depth = plan.depth;
		coef = plan.coef;
		in_object = plan.in_object;
		return true;
	}
	__device__ bool isNotEmpty(uint stack_id) {
		return !planner[stack_id].isEmpty();
	}
};