#include "CudaUtils.cuh"
#include <iostream>

CudaSynchronizer CudaSynchronizer::instance = CudaSynchronizer();

void check_cuda(cudaError_t result, char const* const func, const char* const file, int const line) {
	if (result) {
		std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
			file << ":" << line << " '" << func << "' \n";
		// Make sure we call CUDA Device Reset before exiting
		cudaDeviceReset();
		exit(99);
	}
}

int calculate_max_blocks_number(int threads_per_block, void* kernel_func) {
	cudaFuncAttributes func_attr;
	cudaFuncGetAttributes(&func_attr, kernel_func);

	cudaDeviceProp device_prop;
	cudaGetDeviceProperties(&device_prop, 0);

	int max_threads_per_sm = device_prop.maxThreadsPerMultiProcessor;
	int max_blocks_per_sm = device_prop.maxBlocksPerMultiProcessor;
	int regs_per_sm = device_prop.regsPerMultiprocessor;

	int regs_per_thread = func_attr.numRegs;

	int blocks_by_threads = max_threads_per_sm / threads_per_block;

	int total_regs_per_block = threads_per_block * regs_per_thread;
	int blocks_by_regs = total_regs_per_block > 0 ? regs_per_sm / total_regs_per_block : max_blocks_per_sm;

	int blocks_per_sm = std::min({ blocks_by_threads, blocks_by_regs, max_blocks_per_sm });

	int total_sms = device_prop.multiProcessorCount;
	int total_blocks = blocks_per_sm * total_sms;

	return total_blocks;
}

dim3 calculate_max_blocks_number(dim3 threads_per_block, int work_size_x, int work_size_y, void* kernel_func) {
	cudaFuncAttributes attr;
	cudaFuncGetAttributes(&attr, kernel_func);

	cudaDeviceProp device_prop;
	cudaGetDeviceProperties(&device_prop, 0);

	int max_threads_per_block = device_prop.maxThreadsPerBlock;

	int multi_processor_count = device_prop.multiProcessorCount;

	int blocks_per_sm = device_prop.maxThreadsPerMultiProcessor / (threads_per_block.x * threads_per_block.y);

	int max_blocks = blocks_per_sm * multi_processor_count;

	dim3 grid_dim;
	grid_dim.x = (work_size_x + threads_per_block.x - 1) / threads_per_block.x;
	grid_dim.y = (work_size_y + threads_per_block.y - 1) / threads_per_block.y;
	grid_dim.z = 1;

	if (grid_dim.x * grid_dim.y > max_blocks) {
		grid_dim.y = max_blocks / grid_dim.x;
		if (grid_dim.y == 0) grid_dim.y = 1;
	}

	return grid_dim;
}