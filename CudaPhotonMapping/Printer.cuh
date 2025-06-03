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

	__host__ __device__ static void index_out_of_bound_error(char* error_location) {
		printf("Error in %s. Message: index out of bound\n", error_location);
	}

	__host__ __device__ static void deep_look_stack_error(char* error_content) {
		printf("Error in class DeepLookStack. Message: %s\n", error_content);
	}
	__host__ __device__ static void stack_error(char* error_content) {
		printf("Error in class Stack. Message: %s\n", error_content);
	}

	__host__ static void cuda_properties() {
		auto printer = Printer();

		cudaDeviceProp cudaDeviceProp;
		cudaGetDeviceProperties(&cudaDeviceProp, 0);
		printer.s("Cuda Device Properties:").nl();
		printer.s("   ").s("Compute capabilty ").i(cudaDeviceProp.major).s(".").i(cudaDeviceProp.minor).nl();
		printer.s("   ").s("Max shared memory per block ").i(cudaDeviceProp.sharedMemPerBlock).nl();
		printer.s("   ").s("Reserved shared memory per block ").i(cudaDeviceProp.reservedSharedMemPerBlock).nl();

		size_t free_mem, total_mem;
		cudaError_t err = cudaMemGetInfo(&free_mem, &total_mem);
		free_mem  /= 1024.0 * 1024.0;
		total_mem /= 1024.0 * 1024.0;
		printer.s("   ").s("Total global memory ").i(total_mem).s("mb").nl();
		printer.s("   ").s("Free global memory ").i(free_mem).s("mb").nl();

		size_t heap_size;
		err = cudaDeviceGetLimit(&heap_size, cudaLimitMallocHeapSize);
		heap_size /= 1024.0 * 1024.0;
		printer.s("   ").s("Current heap size ").i(heap_size).s("mb").nl();

		int num_sms, max_threads_per_sm, max_blocks_per_sm, max_threads_per_block;
		cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, 0);
		cudaDeviceGetAttribute(&max_blocks_per_sm, cudaDevAttrMaxBlocksPerMultiprocessor, 0);
		cudaDeviceGetAttribute(&max_threads_per_sm, cudaDevAttrMaxThreadsPerMultiProcessor, 0);
		cudaDeviceGetAttribute(&max_threads_per_block, cudaDevAttrMaxThreadsPerBlock, 0);
		printer.s("   ").s("SMs count ").i(num_sms).nl();
		printer.s("   ").s("Registers per block ").i(cudaDeviceProp.regsPerBlock).nl();
		printer.s("   ").s("Registers per sm ").i(cudaDeviceProp.regsPerMultiprocessor).nl();
		printer.s("   ").s("Max blocks per sm ").i(max_blocks_per_sm).nl();
		printer.s("   ").s("Max threads per sm ").i(max_threads_per_sm).nl();
		printer.s("   ").s("Max threads per block ").i(max_threads_per_block).nl();

		printf("   Cooperative launch %s supported\n\n", cudaDeviceProp.cooperativeLaunch ? "is" : "is NOT");
		//printf("   Multi-device cooperative launch %s supported\n", cudaDeviceProp.cooperativeMultiDeviceLaunch ? "is" : "is NOT");
	}

	__host__ static void kernel_properties(void* func) {
		cudaFuncAttributes attr;
		cudaFuncGetAttributes(&attr, func);
		printf("Registers per thread: %d\n", attr.numRegs);
		printf("Shared memory: %zu\n", attr.sharedSizeBytes);
		printf("Const size: %zu\n", attr.constSizeBytes);
		printf("Local memory per thread: %zu\n", attr.localSizeBytes);
	}
};