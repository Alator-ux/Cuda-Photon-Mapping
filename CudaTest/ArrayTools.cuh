#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

template<typename T, typename Lambda>
__host__ __device__ void array_foreach(T arr[], int from, int to, Lambda lambda) {
	for (int i = from; i < to; i++) {
		lambda(arr[i]);
	}
}