#include "Drawer.cuh"
#include "Defines.cuh"

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

__global__ void gpu_kernel(uchar3* canvas, Raytracer* raytracer, int width, int height) {
	raytracer->render_gpu(canvas, width, height);
}

void Drawer::draw_in_gpu(int frame) {
	size_t size;
	checkCudaErrors(cudaGraphicsMapResources(1, &cuda_resource, 0));
	checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void**)&gpu_canvas, &size, cuda_resource));

	/*dim3 block(16, 16);
	dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
	int shared_memory = block.x * block.y * sizeof(Raytracer::IntersectionInfo);*/

	int threads = THREADS_NUMBER;
	int blocks = (width * height + threads - 1) / threads;
	int shared_memory = threads * sizeof(cpm::vec3*) * 3;
	timer.startCUDA();
	//compute << <blocks, threads >> > (gpu_canvas, width, height, frame);
	gpu_kernel << <blocks, threads, shared_memory >> > (gpu_canvas, gpu_raytracer, width, height);
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

__global__ void gpu_initialize_kernel(Raytracer* raytracer, Scene* scene, RaytracePlanner* planner) {
	if (scene != nullptr) {
		raytracer->set_scene(scene);
	}
	if (planner != nullptr) {
		raytracer->set_planner(planner);
	}
}

void Drawer::initialize_raytracer() {
	int size = width * height; // 786 432
	int max_depth = GlobalParams::max_depth();
	int max_medium_depth = (max_depth + 1) / 2;
	
	cpu_raytracer->initialize_cpu(size, max_depth, max_medium_depth);

	cudaMalloc(&gpu_raytracer, sizeof(Raytracer));
	auto gpu_planner = RaytracePlanner::initialize_gpu(size, max_depth, max_medium_depth, 1.f);
	
	gpu_initialize_kernel << <1, 1 >> > (gpu_raytracer, nullptr, gpu_planner);
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
	// TODO free resources?
	this->cuda_resource = other.cuda_resource;
	this->width = other.width;
	this->height = other.height;
	this->render_mode = other.render_mode;
	this->gpu_canvas = other.gpu_canvas;
	this->cpu_canvas = other.cpu_canvas;
	this->cpu_raytracer = other.cpu_raytracer;
	this->gpu_raytracer = other.gpu_raytracer;
	return (*this);
}