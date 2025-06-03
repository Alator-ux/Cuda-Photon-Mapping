#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <crt/host_defines.h>
#include <cuda_runtime.h>

void check_cuda(cudaError_t result, char const* const func, const char* const file, int const line);

class CudaSynchronizer {
	static CudaSynchronizer instance;
	cudaEvent_t cu_event;
public:
	CudaSynchronizer() {
		cudaEventCreate(&cu_event);
	}
	~CudaSynchronizer() {
		cudaEventDestroy(cu_event);
	}
	void synchronize() {
		cudaEventRecord(cu_event, 0);
		cudaEventSynchronize(cu_event);
	}
	static void synchronize_with_instance() {
		instance.synchronize();
	}
};

namespace cpm {
	template <typename T>
	__host__ __device__ __inline__
	void swap(T& first, T& second) {
		T temp = first;
		first = second;
		second = temp;
	}
}

int calculate_max_blocks_number(int threads_per_block, void* kernel_func);

dim3 calculate_max_blocks_number(dim3 threads_per_block, int work_size_x, int work_size_y, void* kernel_func);