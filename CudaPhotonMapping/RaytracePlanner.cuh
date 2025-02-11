#pragma once
#include <crt/host_defines.h>
#include <cuda_runtime.h>
#include "MediumManager.cuh"
#include "Ray.cuh"

class RaytracePlanner {
public:
	struct RayPlan {
		cpm::Ray ray;
		int depth;
		cpm::vec3 coef;
		bool in_object;
	};
private:
	size_t array_size;
	MediumManager* medium_managers = nullptr;
	cpm::stack<RayPlan>* planner = nullptr;
public:
	__host__ void intialize_cpu(size_t pixels_number, int max_depth, int max_medium_depth) {
		this->array_size = pixels_number;
		/* 786 432 (pixels)
		*		cpm::stack<RayPlan> planner
		*		4 + 4 (size, capacity) + 4 (pointer) + 8 (stack capacity) * (6 * 4 (ray) + 4 (depth) + 3 * 4 (coef)) = 332
		*				
		* 
		*		MediumManager medium_managers
		*		620
		*	
		*		total
		*		332 + 620 = 952
		* total
		* 786 432 * 952 = 748 683 264 byte = 714 mb
		*/

		/* 786 432 (pixels)
		*		cpm::stack<RayPlan> planner
		*		24 + 8 * 64 = 536
		*
		*		MediumManager medium_managers
		*		728
		*
		*		total
		*		332 + 620 = 1 264
		* total
		* 786 432 * 1 264 = 748 683 264 byte = 714 mb
		*/

		cpm::Ray default_ray = cpm::Ray(cpm::vec3(0.f), cpm::vec3(0.f));
		planner = new cpm::stack<RayPlan>[array_size];
		medium_managers = new MediumManager[array_size];
		for (size_t i = 0; i < array_size; i++) {
			planner[i].initialize(max_depth);
			medium_managers[i].intialize(max_depth, max_medium_depth, 1.f);
		}
		
	}
	__device__ void _initialize_device(cpm::stack<RaytracePlanner::RayPlan>* planner, MediumManager* medium_managers, size_t array_size);
	__device__ void _initialize_device_data(cpm::stack<RaytracePlanner::RayPlan>* planner, RaytracePlanner::RayPlan* planner_data,
		MMInnerContainer* medium_manager_inner_container, MMInnerData* medium_manager_innder_data,
		MediumManager* medium_managers, size_t array_size, int max_depth, int max_medium_depth);
	static __host__ RaytracePlanner* initialize_gpu(size_t array_size, int max_depth, int max_medium_depth, float default_refr_index);

	__host__ __device__ RaytracePlanner& operator=(const RaytracePlanner& other)
	{
		if (this == &other)
			return *this;

		this->medium_managers = other.medium_managers;
		this->planner = other.planner;
		this->array_size = other.array_size;

		return *this;
	}

	__host__ __device__ cpm::Tuple3<float, float, bool> get_refractive_indices(size_t stack_id, int model_id, float model_refractive_index) {
		return medium_managers[stack_id].get_refractive_indices(model_id, model_refractive_index);
	}
	__host__ __device__ void push_refraction(size_t stack_id, bool& replace_medium,
		const cpm::Ray& ray, int depth, const cpm::vec3& coef,
		int model_id, float model_refractive_index, float model_opaque, bool in_object) {
		if (stack_id >= array_size) {
			printf("Stack id >= array size in RaytracePlanner, function push_refraction");
			return;
		}
		medium_managers[stack_id].increase_depth(replace_medium);
		medium_managers[stack_id].inform(model_id, model_refractive_index);

		planner[stack_id].push({ ray, depth + 1, coef * (1.f - model_opaque), in_object });

	}
	__host__ __device__ bool pop_refraction(size_t stack_id, bool& replace_medium, 
		cpm::Ray& ray, int& depth, cpm::vec3& coef, bool& in_object) {
		if (replace_medium) { // no refractions by this ray
			medium_managers[stack_id].reduce_depth();
		}
		if (stack_id >= array_size) {
			printf("Stack id >= array size in RaytracePlanner, function pop_refraction");
			return false;
		}
		if (planner[stack_id].isEmpty()) {
			return false;
		}
		RayPlan plan = planner[stack_id].pop();
		ray = plan.ray;
		depth = plan.depth;
		coef = plan.coef;
		in_object = plan.in_object;
		replace_medium = true;
		return true;
	}
	__device__ bool isNotEmpty(size_t stack_id) {
		return !planner[stack_id].isEmpty();
	}
};