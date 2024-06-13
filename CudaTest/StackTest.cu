#include "Test.cuh"

__host__ __device__ void StackTestCore(cpm::stack<int>* stack, int* vres, bool* bres) {
	bres[0] = stack->isFull(); // false
	bres[1] = stack->isEmpty(); // true

	stack->push(333);
	vres[0] = stack->top(); // 333

	stack->push(7);
	vres[1] = stack->top(); // 7
	bres[2] = stack->isEmpty(); // false
	bres[3] = stack->isFull(); // false

	stack->push(2);
	vres[2] = stack->top(); // 2

	stack->pop();
	vres[3] = stack->top(); // 7

	stack->push(3);
	stack->push(4);
	stack->push(5);
	stack->push(6);
	vres[4] = stack->top(); // 4 (if cap == 5)
	bres[4] = stack->isFull(); // true
}

__device__ void StackDeviceTestCore(size_t cap, int* vres, bool* bres) {
	auto s = cpm::stack<int>(cap);
	StackTestCore(&s, vres, bres);
}

__global__ void StackTestFromDevice(size_t cap, int* vres, bool* bres) {
	StackDeviceTestCore(cap, vres, bres);
}

void ctest::StackTest() {
	println_divider();
	std::cout << "Stack Test" << std::endl;
	const size_t res_size = 5;

	int vres_host[res_size];
	int vexpected[] = { 333, 7, 2, 7, 4 };
	bool vmatched[res_size];

	bool bres_host[res_size];
	bool bexpected[] = { false, true, false, false, true };
	bool bmatched[res_size];

	int* vres_device;
	bool* bres_device;
	cudaMalloc((void**)&vres_device, res_size * sizeof(int));
	cudaMalloc((void**)&bres_device, res_size * sizeof(bool));

	size_t c_cap = 5;

	StackTestFromDevice << <1, 1 >> > (c_cap, vres_device, bres_device);
	auto cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "Ошибка синхронизации устройства: %s\n", cudaGetErrorString(cudaStatus));
		exit(EXIT_FAILURE);
	}

	cudaMemcpy(vres_host, vres_device, res_size * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(bres_host, bres_device, res_size * sizeof(bool), cudaMemcpyDeviceToHost);

	is_matched(res_size, vexpected, vres_host, vmatched);
	is_matched(res_size, bexpected, bres_host, bmatched);

	std::cout << "Expected and result values:" << std::endl;
	println_array(vexpected, res_size);
	println_array(vres_host, res_size);
	println_array(vmatched, res_size);

	std::cout << "Expected and result booleans:" << std::endl;
	println_array(bexpected, res_size);
	println_array(bres_host, res_size);
	println_array(bmatched, res_size);

	cudaFree(vres_device);
	cudaFree(bres_device);
	println_divider();
}