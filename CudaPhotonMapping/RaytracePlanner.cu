#include "RaytracePlanner.cuh"
#include "CudaUtils.cuh"
#include "Defines.cuh"
#include "Array.cuh"

__device__ void RaytracePlanner::initialize(cpm::rayplan_stack* planner, MediumManager * medium_managers, uint array_size) {
	this->array_size = array_size;
	this->planner = planner;
	this->medium_managers = medium_managers;
}

__global__ void initialize_ray_planner_kernel(RaytracePlanner* ray_planner,
	cpm::rayplan_stack* planner, MediumManager* medium_managers, uint array_size) {

	ray_planner->initialize(planner, medium_managers, array_size);
}

__host__ cpm::pair<RaytracePlanner*, RaytracePlanner*> RaytracePlanner::initialize(uint array_size, int max_depth, int max_medium_depth, float default_refr_index) {
	uint size = array_size;

	/* GPU */
	cpm::Array<MediumManager> medium_managers;
	medium_managers.initialize_on_device(size);
	medium_managers.fill([](idxtype i) { return MediumManager(); });
	/* CPU */
	MediumManager* cpu_medium_manager = new MediumManager[1];
	cpu_medium_manager[0] = MediumManager();

	/* GPU */
	cpm::Array<cpm::rayplan_stack> planner;
	planner.initialize_on_device(size);
	planner.fill([](idxtype i) {return cpm::rayplan_stack(); });
	/* CPU */
	cpm::rayplan_stack* cpu_planner = new cpm::rayplan_stack[1];
	cpu_planner[0] = cpm::rayplan_stack();
	
	/* GPU */
	size *= max_depth;
	RayPlan* planner_data;
	cudaMalloc(&planner_data, sizeof(RayPlan) * size);
	/* CPU */
	RayPlan* cpu_planner_data = new RayPlan[max_depth];

	/* GPU */
	MMInnerContainer* medium_manager_inner_container;
	cudaMalloc(&medium_manager_inner_container, sizeof(MMInnerContainer) * size);
	cudaMemset(medium_manager_inner_container, 0, sizeof(MMInnerContainer) * size);
	/* CPU */
	MMInnerContainer* cpu_medium_manager_inner_container = new MMInnerContainer[max_depth];

	/* GPU */
	size *= max_medium_depth;
	MMInnerData* medium_manager_innder_data;
	cudaMalloc(&medium_manager_innder_data, sizeof(MMInnerData) * size);
	cudaMemset(medium_manager_innder_data, 0, sizeof(MMInnerData) * size);
	/* CPU */
	MMInnerData* cpu_medium_manager_innder_data = new MMInnerData[max_depth * max_medium_depth];

	/* GPU */
	RaytracePlanner* ray_planner;
	cudaMalloc(&ray_planner, sizeof(RaytracePlanner));

	initialize_ray_planner_kernel << <1, 1 >> > (ray_planner, planner.get_data(), medium_managers.get_data(), array_size);
	CudaSynchronizer::synchronize_with_instance();
	checkCudaErrors(cudaGetLastError());
	/* CPU */
	RaytracePlanner* cpu_ray_planner = new RaytracePlanner();
	cpu_ray_planner->initialize(cpu_planner, cpu_medium_manager, 1);

	GlobalParams::set_medium_manager_parameters(
		cpu_medium_manager_innder_data, medium_manager_innder_data, max_medium_depth,
		cpu_medium_manager_inner_container, medium_manager_inner_container, max_depth,
		cpu_planner_data, planner_data, max_depth);

	return { cpu_ray_planner, ray_planner };
}