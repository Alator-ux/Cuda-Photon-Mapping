#pragma once
#include <GL/glew.h>
#include <cuda_gl_interop.h>
#include "CudaUtils.cuh"
#include <iostream>

void create_pbo(GLuint* pbo, cudaGraphicsResource** cgr, unsigned int size_bytes, unsigned int flags);

void delete_pbo(GLuint* pbo, cudaGraphicsResource* cgr);

void create_texture(GLuint* texture, int width, int height);

void delete_texture(GLuint* texture);

void checkOpenGLErrors();