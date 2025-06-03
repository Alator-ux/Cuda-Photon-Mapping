#include "Drawer.cuh"
#include "Defines.cuh"
#include "PhotonMaxHeap.cuh"
#include "CudaUtils.cuh"

__global__ void compute(uchar3* canvas, int width, int height, int frame) {
	/*int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x >= width || y >= height) return;*/
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id >= width * height) return;
	int x = id % width;
	int y = id / width;

	canvas[id] = make_uchar3((x + frame) % 256, (y + frame) % 256, 128);
}

__global__ void gpu_kernel(uchar3* canvas, Raytracer* raytracer, int max_working_threads, int width, int height) {
	raytracer->render_gpu(canvas, max_working_threads, width, height);
}

void Drawer::draw_in_gpu(int frame) {
	size_t size;
	checkCudaErrors(cudaGraphicsMapResources(1, &cuda_resource, 0));
	checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void**)&gpu_canvas, &size, cuda_resource));

	/*dim3 threads(TWO_D_THREADS_NUMBER, TWO_D_THREADS_NUMBER);
	dim3 blocks((width + threads.x - 1) / threads.x, (height + threads.y - 1) / threads.y);*/
	/*int shared_memory = block.x * block.y * sizeof(Raytracer::IntersectionInfo);*/

	/*int threads = THREADS_NUMBER;
	int blocks = (width * height + threads - 1) / threads;
	int shared_memory = threads * sizeof(cpm::vec3*) * 3;*/
	timer.startCUDA();
	//compute << <blocks, threads >> > (gpu_canvas, width, height, frame);
	gpu_kernel << <blocks_2d, threads_2d >> > (gpu_canvas, gpu_raytracer, max_working_threads, width, height);
	timer.stopCUDA();
	timer.printCUDA();
	checkCudaErrors(cudaGetLastError());
	cudaGraphicsUnmapResources(1, &cuda_resource, 0);
}

void Drawer::draw_in_cpu(int frame) {
	timer.startCPU();
	cpu_raytracer->render_cpu(cpu_canvas);
	timer.stopCPU();
	timer.printCPU();
}

void Drawer::Draw(int frame) {
	if (render_mode == RenderMode::cpu) {
		draw_in_cpu(frame);
	}
	else {
		draw_in_gpu(frame);
	}
}

__global__ void gpu_initialize_kernel(Raytracer* raytracer, Scene* scene, RaytracePlanner* planner, PhotonMaxHeap* heaps = nullptr) {
	if (scene != nullptr) {
		raytracer->set_scene(scene);
	}
	if (planner != nullptr) {
		raytracer->set_planner(planner);
	}
	if (heaps != nullptr) {
		raytracer->set_heap_pointer(heaps);
	}
}

void Drawer::initialize_raytracer() {
	int pixels = width * height;
	int max_depth = GlobalParams::max_depth();
	int max_medium_depth = (max_depth + 1) / 2;

	threads_2d = { TWO_D_THREADS_NUMBER, TWO_D_THREADS_NUMBER, 1 };
	blocks_2d = { (width + threads_2d.x - 1) / threads_2d.x, (height + threads_2d.y - 1) / threads_2d.y, 1 };
	dim3 max_blocks = calculate_max_blocks_number(threads_2d, width, height, gpu_kernel);
	blocks_2d.x = std::min(blocks_2d.x, max_blocks.x);
	blocks_2d.y = std::min(blocks_2d.y, max_blocks.y);
	max_working_threads = blocks_2d.x * blocks_2d.y * threads_2d.x * threads_2d.y;

	cudaMalloc(&gpu_raytracer, sizeof(Raytracer));
	auto planners = RaytracePlanner::initialize(max_working_threads, max_depth, max_medium_depth, GlobalParams::default_refractive_index());

	// Photon Max Heap intialization
	int heap_capacity = std::max(GlobalParams::global_photon_num(), GlobalParams::caustic_photon_num());
	auto cpu_photon_heap = new PhotonMaxHeap();
	auto cpu_photon_heap_data = new PhotonMaxHeapItem[heap_capacity];
	
	cpu_raytracer->set_planner(planners.first);
	cpu_raytracer->set_heap_pointer(cpu_photon_heap);

	// Photon Max Heap intialization
	PhotonMaxHeap* gpu_photon_heap;
	cudaMalloc(&gpu_photon_heap, sizeof(PhotonMaxHeap) * max_working_threads);
	cudaMemset(gpu_photon_heap, 0, sizeof(PhotonMaxHeap) * max_working_threads);
	PhotonMaxHeapItem* gpu_photon_heap_data;
	cudaMalloc(&gpu_photon_heap_data, sizeof(PhotonMaxHeapItem) * max_working_threads * heap_capacity);
	cudaMemset(gpu_photon_heap_data, 0, sizeof(PhotonMaxHeapItem) * max_working_threads * heap_capacity);
	set_photon_heap_parameters(cpu_photon_heap_data, gpu_photon_heap_data, heap_capacity);

	gpu_initialize_kernel << <1, 1 >> > (gpu_raytracer, nullptr, planners.second, gpu_photon_heap);
	CudaSynchronizer::synchronize_with_instance();
	checkCudaErrors(cudaGetLastError());


	Printer().s("Raytracer initialized").nl();
	Printer::cuda_properties();
}

void Drawer::change_render_mode() {
	set_render_mode(render_mode == RenderMode::cpu ? RenderMode::gpu : RenderMode::cpu);
}

void Drawer::set_render_mode(RenderMode render_mode) {
	if (render_mode == RenderMode::cpu) {

	}
	this->render_mode = render_mode;
}

void Drawer::set_scenes(Scene* cpu_scene, Scene* gpu_scene) {
	cpu_raytracer->set_scene(cpu_scene);
	gpu_initialize_kernel << <1, 1 >> > (gpu_raytracer, gpu_scene, nullptr);
	cudaEvent_t cu_event;
	cudaEventCreate(&cu_event);
	cudaEventSynchronize(cu_event);
	checkCudaErrors(cudaGetLastError());
	Printer().s("Scene updated").nl();
}

__global__ void gpu_initialize_photon_maps_kernel(Raytracer* raytracer, PhotonGrid diffuse_grid, PhotonGrid specular_grid) {
	raytracer->set_photon_maps(diffuse_grid, specular_grid);
}

void Drawer::set_photon_maps(cpm::four_tuple<PhotonGrid, PhotonGrid, PhotonGrid, PhotonGrid> photon_maps) {
	/*cpu_raytracer->set_photon_maps(diff)*/
	gpu_initialize_photon_maps_kernel << <1, 1 >> > (gpu_raytracer, photon_maps.third, photon_maps.fourth);
	CudaSynchronizer::synchronize_with_instance();
	checkCudaErrors(cudaGetLastError());

	cpu_raytracer->set_photon_maps(photon_maps.first, photon_maps.second);
}

RenderMode Drawer::get_render_mode() {
	return render_mode;
}

uchar3* Drawer::get_cpu_canvas() {
	return cpu_canvas;
}

Drawer& Drawer::operator=(const Drawer& other) {
	if (this == &other) {
		return (*this);
	}
	
	this->cuda_resource = other.cuda_resource;
	this->width = other.width;
	this->height = other.height;
	this->render_mode = other.render_mode;
	this->gpu_canvas = other.gpu_canvas;
	this->cpu_canvas = other.cpu_canvas;
	this->cpu_raytracer = other.cpu_raytracer;
	this->gpu_raytracer = other.gpu_raytracer;
	this->max_working_threads = other.max_working_threads;
	this->threads_2d = other.threads_2d;
	this->blocks_2d = other.blocks_2d;
	return (*this);
}