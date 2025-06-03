#pragma once
#include "CudaUtils.cuh"
#include "Raytracer.cuh"
#include <GL/glew.h>
#include "RenderEnums.cuh"
#include "Timer.cuh"
#include "PhotonGrid.cuh"
#include "FourTuple.cuh"

class Drawer {
	RenderMode render_mode;
	uchar3* gpu_canvas;
	uchar3* cpu_canvas;
	cudaGraphicsResource* cuda_resource;
	int width, height;
	int max_working_threads;
	dim3 threads_2d;
	dim3 blocks_2d;
	Raytracer* cpu_raytracer;
	Raytracer* gpu_raytracer;

	Timer timer;

	void draw_in_gpu(int frame);
	void draw_in_cpu(int frame);
	void initialize_raytracer();
public:
	Drawer() : gpu_canvas(nullptr), cpu_canvas(nullptr), cuda_resource(nullptr), width(0), height(0) {}
	Drawer(const Drawer& other) : 
		cuda_resource(other.cuda_resource), width(other.width), height(other.height), render_mode(other.render_mode),
		gpu_canvas(other.gpu_canvas), cpu_canvas(other.cpu_canvas),
		cpu_raytracer(other.cpu_raytracer), gpu_raytracer(other.gpu_raytracer),
		max_working_threads(other.max_working_threads), threads_2d(other.threads_2d), blocks_2d(other.blocks_2d),
		timer() {}
	Drawer(cudaGraphicsResource* cuda_resource, int width, int height, RenderMode render_mode = RenderMode::gpu)
		: cuda_resource(cuda_resource), width(width), height(height), render_mode(render_mode),
		gpu_canvas(nullptr), cpu_canvas((uchar3*)malloc(width * height * sizeof(uchar3))),
		cpu_raytracer((Raytracer*)malloc(sizeof(Raytracer))),
		timer()
	{
		initialize_raytracer();
	}

	Drawer& operator=(const Drawer& other);

	void Draw(int frame);
	void change_render_mode();
	void set_render_mode(RenderMode render_mode);
	void update_camera(Camera camera) { }
	void set_scenes(Scene* cpu_scene, Scene* gpu_scene);
	void set_photon_maps(cpm::four_tuple<PhotonGrid, PhotonGrid, PhotonGrid, PhotonGrid> photon_maps);
	RenderMode get_render_mode();
	uchar3* get_cpu_canvas();
};