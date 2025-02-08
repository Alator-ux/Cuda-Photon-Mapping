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
		printf("Error in class DeepLookStack. Message: %s\n", error_content);
	}
	__host__ __device__ static void stack_error(char* error_content) {
		printf("Error in class Stack. Message: %s\n", error_content);
	}

	__host__ __device__ static void cuda_properties() {
		auto printer = Printer();

		cudaDeviceProp cudaDeviceProp;
		cudaGetDeviceProperties(&cudaDeviceProp, 0);
		printer.s("Cuda Device Properties:").nl();
		printer.s("   ").s("Max shared memory per block ").i(cudaDeviceProp.sharedMemPerBlock).nl();
		printer.s("   ").s("Reserved shared memory per block ").i(cudaDeviceProp.reservedSharedMemPerBlock).nl();

		size_t free_mem, total_mem;
		cudaError_t err = cudaMemGetInfo(&free_mem, &total_mem);
		free_mem  /= 1024.0 * 1024.0;
		total_mem /= 1024.0 * 1024.0;
		printer.s("   ").s("Total global memory ").i(total_mem).s("mb").nl();
		printer.s("   ").s("Free global memory ").i(free_mem).s("mb").nl();
	}
};