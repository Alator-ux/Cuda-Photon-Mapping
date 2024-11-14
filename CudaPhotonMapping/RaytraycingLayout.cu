#include "Test.cuh"
#include "vec3.h"
#define __CUDA_ARCH__
void RTLayoytTest() {
	const int threadsPerBlock = 32;

	const int width = 800;
	const int height = 600;
	int totalPixels = width * height;
	int blocks = (totalPixels + threadsPerBlock - 1) / threadsPerBlock;

	vec3* pixels;

}