#include "CudaOpenGLUtils.cuh"
#include "Defines.cuh"

void create_pbo(GLuint* pbo, cudaGraphicsResource** cgr, unsigned int size_bytes, unsigned int flags) {
	glGenBuffers(1, pbo);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, *pbo);
	glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB, size_bytes, NULL, GL_STREAM_DRAW); // GL_STREAM_DRAW

	checkCudaErrors(cudaGraphicsGLRegisterBuffer(cgr, *pbo, flags));
}

void delete_pbo(GLuint* pbo, cudaGraphicsResource* cgr) {
	checkCudaErrors(cudaGraphicsUnregisterResource(cgr));

	glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, *pbo);
	glDeleteBuffers(1, pbo);

	*pbo = 0;
}

void create_texture(GLuint* texture, int width, int height) {
	glGenTextures(1, texture);
	glBindTexture(GL_TEXTURE_2D, *texture);

	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB8, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, nullptr);

	glBindTexture(GL_TEXTURE_2D, 0);
}

void delete_texture(GLuint* texture) {
	glDeleteTextures(1, texture);
	*texture = 0;
}

void checkOpenGLErrors() {
	GLenum errCode;
	// Коды ошибок можно смотреть тут
	// https://www.khronos.org/opengl/wiki/OpenGL_Error
	if ((errCode = glGetError()) != GL_NO_ERROR) {
		std::cout << "OpenGl error! (" << errCode << "): " << glewGetErrorString(errCode) << std::endl;
	}
}