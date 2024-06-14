#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

namespace cpm
{
	template<typename FirstType, typename SecondType>
	struct pair {
		FirstType first;
		SecondType second;
		__host__ __device__ pair(FirstType first, SecondType second) {
			this->first = first;
			this->second = second;
		}
		__host__ __device__ pair() {
			first = FirstType();
			second = SecondType();
		}
	};
}