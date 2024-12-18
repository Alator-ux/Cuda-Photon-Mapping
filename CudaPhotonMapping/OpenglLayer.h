#pragma once
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include "CudaOpenGLUtils.cuh"

class OpenglLayer {
	GLuint shader_program;
	GLuint vao, vbo;
	int width, height;

	GLuint create_shader_program(const std::string& vertex_shader_path, const std::string& fragment_shader_path);

	void create_vao() {
		float vertices[] = {
			// Позиции       // Текстурные координаты
			-1.0f, -1.0f,    0.0f, 0.0f, // Левый нижний угол
			 1.0f, -1.0f,    1.0f, 0.0f, // Правый нижний угол
			 1.0f,  1.0f,    1.0f, 1.0f, // Правый верхний угол
			-1.0f,  1.0f,    0.0f, 1.0f  // Левый верхний угол
		};

		glGenVertexArrays(1, &vao);
		glGenBuffers(1, &vbo);

		glBindVertexArray(vao);

		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
		glEnableVertexAttribArray(0);

		glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(2 * sizeof(float)));
		glEnableVertexAttribArray(1);

		glBindBuffer(GL_ARRAY_BUFFER, 0);
		glBindVertexArray(0);
	}
public:
	GLuint texture;

	OpenglLayer(): shader_program(0), texture(0), vao(0), vbo(0), width(0), height(0) {}
	OpenglLayer(const std::string& vertex_shader_path, const std::string& fragment_shader_path, int width, int height) 
		: width(width), height(height) {
		shader_program = create_shader_program(vertex_shader_path, fragment_shader_path);
		// create_texture(&texture, width, height); // idk
		create_vao();
	}

	void transfer_pbo_to_texture(GLuint pbo) {
		glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
		glBindTexture(GL_TEXTURE_2D, texture);

		glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE, nullptr);

		glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
		glBindTexture(GL_TEXTURE_2D, 0);
		checkOpenGLErrors();
	}

	void transfer_array_to_texture(uchar3* canvas) {
		glBindTexture(GL_TEXTURE_2D, texture);

		glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE, canvas);

		glBindTexture(GL_TEXTURE_2D, 0);
		checkOpenGLErrors();
	}

	void draw() {
		glUseProgram(shader_program);

		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, texture);
		glUniform1i(glGetUniformLocation(shader_program, "textureSampler"), 0);

		glBindVertexArray(vao);
		glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
		glBindVertexArray(0);

		glBindTexture(GL_TEXTURE_2D, 0);
		glUseProgram(0);
		checkOpenGLErrors();
	}

	~OpenglLayer() {
		delete_texture(&texture);
	}
};