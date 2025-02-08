#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

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