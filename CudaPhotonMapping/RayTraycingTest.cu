#include "Test.cuh"

__device__ cpm::vec3 color(const cpm::Ray ray) {
	float t = 0.5f * (ray.direction.y + 1.f);
	return (1.0f - t) * cpm::vec3(1.0, 1.0, 1.0) + t * cpm::vec3(0.5, 0.7, 1.0);
}

__global__ void render(cpm::vec3* frame_buffer, int fb_width, int fb_height) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	if ((x >= fb_width) || (y >= fb_height)) {
		return;
	}
	int pixel_index = x + y * fb_width;

	auto a = cpm::vec3(y, x, 0);
	auto b = cpm::vec3(y*x, y*x*2, 3);
	cpm::Ray c(a,b);
	frame_buffer[pixel_index] = color(c);
}

void print_frame_buffer(cpm::vec3* frame_buffer, int fb_width, int fb_height) {
	std::cout << "P3\n" << fb_width << " " << fb_height << "\n255\n";
	for (int j = fb_height - 1; j >= 0; j--) {
		for (int i = 0; i < fb_width; i++) {
			size_t pixel_index = j * fb_width + i;
			cpm::vec3 rgb = frame_buffer[pixel_index];
			std::cout << rgb.x << " " << rgb.y << " " << rgb.z << "\n";
		}
	}
}

void ctest::RayTracingTest() {
	println_divider();
	std::cout << "Ray Tracing Test" << std::endl;

	int fb_width = 20;
	int fb_height = 20;

	cpm::vec3* frame_buffer;
	checkCudaErrors(cudaMallocManaged(&frame_buffer, fb_width * fb_height * sizeof(cpm::vec3)));

	int thread_width = 8;
	int thread_height = 8;
	dim3 blocks(fb_width / thread_width + 1, fb_height / thread_height);
	dim3 threads(thread_width, thread_height);
	render << <blocks, threads>> > (frame_buffer, fb_width, fb_height);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
	//---------------------------------------------//
	//print_frame_buffer(frame_buffer, fb_width, fb_height);
	checkCudaErrors(cudaFree(frame_buffer));
	
	

	println_divider();
}