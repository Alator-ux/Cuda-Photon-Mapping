#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <cuda_gl_interop.h>
#include "CudaUtils.cuh"
#include "Drawer.cuh"
#include <iostream>
#include "CudaOpenGLUtils.cuh"
#include <string>
#include "OpenglLayer.h"
#include "PhotonGrid.cuh"
#include "FourTuple.cuh"

class Window {
	GLuint pbo;
	cudaGraphicsResource* cuda_resource;
	OpenglLayer opengl_layer;

	Scene* cpu_scene;
	Scene* gpu_scene;

	RenderMode render_mode;

	std::string title;
	GLFWwindow* glfw_window;

	Drawer drawer;
	int width, height;

	int frame_count = 0;
	double previous_time = 0;

	int frame = 0;

	bool render_frame = true;
	bool frame_rendered = false;

	bool space_was_pressed = false;
	bool n_was_pressed = false;
	void process_input() {
		{
			int state = glfwGetKey(glfw_window, GLFW_KEY_SPACE);
			if (state == GLFW_PRESS && !space_was_pressed) {
				if (render_mode == RenderMode::cpu) {
					std::cout << "Render device is GPU" << std::endl;
					render_mode = RenderMode::gpu;
				}
				else {
					std::cout << "Render device is CPU" << std::endl;
					render_mode = RenderMode::cpu;
				}
				drawer.set_render_mode(render_mode);
				space_was_pressed = true;
			}
			if (state == GLFW_RELEASE) {
				space_was_pressed = false;
			}
		}
		{
			int state = glfwGetKey(glfw_window, GLFW_KEY_N);
			if (state == GLFW_PRESS) {
				std::cout << "Start render next frame" << std::endl;
				render_frame = true;
				n_was_pressed = true;
			}
			if (state == GLFW_RELEASE) {
				n_was_pressed = false;
			}
		}
	}

	void initialize_glfw_window() {
		if (!glfwInit()) {
			std::cerr << "Failed to initialize GLFW\n";
			exit(-1);
		}

		glfw_window = glfwCreateWindow(width, height, title.c_str(), nullptr, nullptr);
		if (!glfw_window) {
			std::cerr << "Failed to create GLFW window\n";
			glfwTerminate();
			exit(-1);
		}

		glfwMakeContextCurrent(glfw_window);
		glewInit();
	}

	void initialize_cuda_part() {
		unsigned int size_bytes = width * height * sizeof(uchar3);

		/*create_pbo(&this->pbo, &this->cuda_resource, size_bytes, cudaGraphicsMapFlagsWriteDiscard);*/
		create_pbo(&this->pbo, &this->cuda_resource, size_bytes, cudaGraphicsMapFlagsWriteDiscard);

		drawer = Drawer(cuda_resource, width, height, render_mode);
	}

	void initialize_opengl_part() {
		glDisable(GL_DEPTH_TEST);
		glDisable(GL_BLEND);
		glClearColor(0.0, 0.0, 0.0, 1.0);
		glClear(GL_COLOR_BUFFER_BIT);
		opengl_layer = OpenglLayer("shaders/main.vert", "shaders/main.frag", width, height);
		create_texture(&opengl_layer.texture, width, height); // idk
	}
	
	void render() {
		if (render_frame) {
			drawer.Draw(frame);
			frame++;

			if (render_mode == RenderMode::gpu) {
				opengl_layer.transfer_pbo_to_texture(pbo);
			}
			else {
				opengl_layer.transfer_array_to_texture(drawer.get_cpu_canvas());
			}
			render_frame = false;
			frame_rendered = true;
		}
		opengl_layer.draw();

		glfwSwapBuffers(glfw_window);
		glfwPollEvents();

		checkOpenGLErrors();
	}

	void calculate_FPS() {
		if (!frame_rendered) {
			return;
		}
		double current_time = glfwGetTime();
		double elapsed_time = current_time - previous_time;

		frame_count++;

		if (elapsed_time < 1.0) {
			return;
		}

		int fps = frame_count / elapsed_time;
		std::string new_title = title + ". FPS: " + std::to_string(fps);
		glfwSetWindowTitle(glfw_window, new_title.c_str());

		frame_count = 0;
		previous_time = current_time;
		frame_rendered = false;
	}

public:

	Window(int width = 400, int height = 400, std::string title = "Window") {
		this->width  = width;
		this->height = height;
		this->title = title;

		render_mode = RenderMode::gpu;

		initialize_glfw_window();
		initialize_cuda_part();
		initialize_opengl_part();
		checkOpenGLErrors();

		
	}

	void Update(){
		if (glfwWindowShouldClose(glfw_window)) {
			exit(0);
		}

		process_input();

		render();
		
		calculate_FPS();
	}

	void set_scenes(Scene* cpu_scene, Scene* gpu_scene) {
		this->cpu_scene = cpu_scene;
		this->gpu_scene = gpu_scene;
		drawer.set_scenes(cpu_scene, gpu_scene);
	}

	void set_maps(cpm::four_tuple<PhotonGrid, PhotonGrid, PhotonGrid, PhotonGrid> photon_maps) {
		drawer.set_photon_maps(photon_maps);
	}

	~Window() {
		delete_pbo(&this->pbo, this->cuda_resource);
		glfwDestroyWindow(glfw_window);
		glfwTerminate();
	}
};