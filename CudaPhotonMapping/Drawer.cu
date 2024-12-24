#include "Drawer.cuh"

__global__ void compute(uchar3* canvas, int width, int height, int frame) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x >= width || y >= height) return;

	int index = y * width + x;
	//canvas[index] = make_uchar4((x + frame) % 256, (y + frame) % 256, 128, 255);
	canvas[index] = make_uchar3((x + frame) % 256, (y + frame) % 256, 128);
}

__global__ void gpu_kernel(uchar3* canvas, Raytracer* raytracer, int width, int height) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x >= width || y >= height) return;

	raytracer->render_gpu(canvas, width, height);
}

void Drawer::draw_in_gpu(int frame) {
	size_t size;
	checkCudaErrors(cudaGraphicsMapResources(1, &cuda_resource, 0));
	checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void**)&gpu_canvas, &size, cuda_resource));

	dim3 block(16, 16);
	dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
	//compute << <grid, block >> > (gpu_canvas, width, height, frame);
	timer.startCUDA();
	gpu_kernel << <grid, block >> > (gpu_canvas, gpu_raytracer, width, height);
	timer.stopCUDA();
	timer.printCUDA();
	/*checkCudaErrors(cudaDeviceSynchronize());*/
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

void Drawer::change_render_mode() {
	set_render_mode(render_mode == RenderMode::cpu ? RenderMode::gpu : RenderMode::cpu);
}

void Drawer::set_render_mode(RenderMode render_mode) {
	if (render_mode == RenderMode::cpu) {

	}
	this->render_mode = render_mode;
}

void Drawer::set_scenes(Scene* cpu_scene, Scene* gpu_scene) {
	Scene* fake_gpu_scene = (Scene*)malloc(sizeof(Scene));
	cudaMemcpy(fake_gpu_scene, gpu_scene, sizeof(Scene), cudaMemcpyDeviceToHost);
	cpu_raytracer->set_scene(fake_gpu_scene);
	cudaMemcpy(gpu_raytracer, cpu_raytracer, sizeof(Raytracer), cudaMemcpyHostToDevice);
	cpu_raytracer->set_scene(cpu_scene);
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