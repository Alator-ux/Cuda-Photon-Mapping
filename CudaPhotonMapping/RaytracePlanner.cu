#include "RaytracePlanner.cuh"
#include "CudaUtils.cuh"
#include "Defines.cuh"

__device__ void RaytracePlanner::_initialize_device_data(cpm::stack<RaytracePlanner::RayPlan>* planner, RaytracePlanner::RayPlan* planner_data,
	MMInnerContainer* medium_manager_inner_container, MMInnerData* medium_manager_innder_data,
	MediumManager* medium_managers, size_t array_size, int max_depth, int max_medium_depth) {
	uint id = blockIdx.x * blockDim.x + threadIdx.x;

	this->planner[id].set_data(planner_data + max_depth * id, max_depth);
	this->medium_managers[id].set_data(medium_manager_inner_container + max_depth * id,
		medium_manager_innder_data + max_depth * max_medium_depth * id, max_depth, max_medium_depth);
	
}

__global__ void initialize_ray_planner_data_kernel(RaytracePlanner* ray_planner, 
	cpm::stack<RaytracePlanner::RayPlan>* planner, RaytracePlanner::RayPlan* planner_data,
	MMInnerContainer* medium_manager_inner_container, MMInnerData* medium_manager_innder_data,
	MediumManager* medium_managers,
	size_t array_size, int max_depth, int max_medium_depth) {

	uint id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id >= array_size) {
		return;
	}
	ray_planner->_initialize_device_data(planner, planner_data, medium_manager_inner_container, medium_manager_innder_data,
		medium_managers, array_size, max_depth, max_medium_depth);
}

__device__ void RaytracePlanner::_initialize_device(cpm::stack<RaytracePlanner::RayPlan>*planner, MediumManager * medium_managers, size_t array_size) {
	this->array_size = array_size;
	this->planner = planner;
	this->medium_managers = medium_managers;
}

__global__ void initialize_ray_planner_kernel(RaytracePlanner* ray_planner,
	cpm::stack<RaytracePlanner::RayPlan>* planner, MediumManager* medium_managers,  size_t array_size) {

	ray_planner->_initialize_device(planner, medium_managers, array_size);
}

__host__ RaytracePlanner* RaytracePlanner::initialize_gpu(size_t array_size, int max_depth, int max_medium_depth, float default_refr_index) {
	size_t size = array_size;

	MediumManager* medium_managers;
	cudaMalloc(&medium_managers, sizeof(MediumManager) * size);

	cpm::stack<RayPlan>* planner;
	cudaMalloc(&planner, sizeof(cpm::stack<RayPlan>) * size);

	size *= max_depth;
	RayPlan* planner_data;
	cudaMalloc(&planner_data, sizeof(RayPlan) * size);
	
	//size += 1;
	MMInnerContainer* medium_manager_inner_container;
	cudaMalloc(&medium_manager_inner_container, sizeof(MMInnerContainer) * size);
	
	//size *= stack_cap + 1;
	size *= max_medium_depth;
	MMInnerData* medium_manager_innder_data;
	cudaMalloc(&medium_manager_innder_data, sizeof(MMInnerData) * size);
	
	RaytracePlanner* ray_planner;
	cudaMalloc(&ray_planner, sizeof(RaytracePlanner));

	initialize_ray_planner_kernel << <1, 1 >> > (ray_planner, planner, medium_managers, array_size);
	CudaSynchronizer::synchronize_with_instance();
	checkCudaErrors(cudaGetLastError());

	int threads = 256;
	int blocks = (array_size + threads - 1) / threads;
	initialize_ray_planner_data_kernel << <blocks, threads >> > (ray_planner, 
		planner, planner_data, 
		medium_manager_inner_container, medium_manager_innder_data,
		medium_managers, 
		array_size, max_depth, max_medium_depth);
	CudaSynchronizer::synchronize_with_instance();
	checkCudaErrors(cudaGetLastError());

	return ray_planner;
}