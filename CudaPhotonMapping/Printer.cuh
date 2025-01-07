#pragma once
#include <crt/host_defines.h>
#include "vec3.cuh"

struct Printer {
	__host__ __device__ Printer() {}
	__host__ __device__ Printer& v3(const cpm::vec3& vec3) {
		printf("(%f, %f, %f)", vec3.x, vec3.y, vec3.z);
		return *this;
	}
	__host__ __device__ Printer& s(const char* str) {
		printf("%s", str);
		return *this;
	}
	__host__ __device__ Printer& i(int integer) {
		printf("%d", integer);
		return *this;
	}
	__host__ __device__ Printer& nl() {
		printf("\n");
		return *this;
	}

	__host__ __device__ static void deep_look_stack_error(char* error_content) {
		printf("\nError in class DeepLookStack. Message: %s\n", error_content);
	}

	__host__ __device__ static void cuda_properties(cudaDeviceProp cudaDeviceProp) {
		auto printer = Printer();
		printer.s("Cuda Device Properties:").nl();
		printer.s("   ").s("Max shared memory per block ").i(cudaDeviceProp.sharedMemPerBlock).nl();
	}
};