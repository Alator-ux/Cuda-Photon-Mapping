#include "CudaRandom.cuh"
#include "CudaUtils.cuh"

namespace cpm {
	__global__ void setup_curand(curandState* states, int size) {
		int id = threadIdx.x + blockIdx.x * blockDim.x;
		if (id < size) {
			curand_init(1234, id, 0, &states[id]);
		}
	}

	cpm::CudaRandom::CudaRandom(int total_threads) {
		this->states_size = total_threads;
		checkCudaErrors(cudaMalloc((void**)&this->states, this->states_size * sizeof(curandState)));

		int threads_per_block = 256;
		int blocks = (total_threads + threads_per_block - 1) / threads_per_block;

		setup_curand << <blocks, threads_per_block >> > (this->states, this->states_size);
		checkCudaErrors(cudaGetLastError());
		checkCudaErrors(cudaDeviceSynchronize());
	}

	cpm::CudaRandom::~CudaRandom() {
		checkCudaErrors(cudaFree(this->states));
	}


	cpm::Random::Random(unsigned int seed) : generator(seed), uniform_distribution(0.f, 1.f) {}
}
