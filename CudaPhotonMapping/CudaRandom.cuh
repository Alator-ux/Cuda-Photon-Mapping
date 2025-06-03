#pragma once
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include "CudaUtils.cuh"
#include <random>
#include "Defines.cuh"

namespace cpm {
	float inline __host__ __device__ __forceinline__ fmap_to_range(float value, float lower, float upper) {
		return lower + value * (upper - lower);
	}

	int inline __host__ __device__ __forceinline__ imap_to_range(float value, int lower, int upper) {
		return roundf(lower + value * (upper - lower));
	}

	struct CudaRandom {
	public:
		cudaRandomStateT* states;
		int states_size;
		CudaRandom(int total_threads);
		~CudaRandom();
	};

	struct Random {
	public:
		std::mt19937 generator;
		std::uniform_real_distribution<float> uniform_distribution;
		Random(unsigned int seed);
		__host__ inline float cpurand_uniform() {
			return uniform_distribution(generator);
		}
		__host__ inline float cpurand_uniform_in_range(float lower, float upper) {
			return fmap_to_range(cpurand_uniform(), lower, upper);
		}
		__host__ inline int cpurand_int_in_range(int lower, int upper) {
			std::uniform_int_distribution<int> distribution(lower, upper);
			return distribution(generator);
		}
	};


}