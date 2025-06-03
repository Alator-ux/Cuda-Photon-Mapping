#pragma once
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <crt/host_defines.h>
#include "Defines.cuh"
#include "GlobalParams.cuh"

namespace {
	__device__ volatile unsigned int grid_synchronizer_count = 0;
};


struct CudaGridSynchronizer {
	__device__ static void synchronize_grid() {
		uint oldc;
		__threadfence();

		if (threadIdx.x == 0)
		{
			oldc = atomicInc((uint*)&grid_synchronizer_count, gridDim.x - 1);
			__threadfence();
			//printf("%i ", blockIdx.x);
			if (oldc != (gridDim.x - 1))
				while (grid_synchronizer_count != 0);
		}
		__syncthreads();
	}
	__device__ static bool need_synchronization() {
		return grid_synchronizer_count > 0;
	}
	__device__ static int get_grid_synchronizer_count() { return grid_synchronizer_count; }
};