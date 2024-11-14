#include "Test.cuh"

struct PQComparator {
	__host__ __device__ bool operator()(const int* f, const int* s) const {
		return (*f) < (*s);
	}
};
void __host__ __device__ TestCore(cpm::priority_queue<int, PQComparator>* q,
	int* vres, bool* bres) {
	bres[0] = q->empty();

	q->push(1);
	vres[0] = q->top();

	q->push(4);
	vres[1] = q->top();
	bres[1] = q->empty();

	q->push(-10);
	vres[2] = q->top();

	q->push(-2);
	vres[3] = q->top();
	bres[2] = q->size() == 4;

	q->pop();
	vres[4] = q->top();
	bres[3] = q->empty();
	bres[4] = q->size() == 3;
}
void __global__ TestFromHostToDevice(cpm::priority_queue<int, PQComparator>* q, 
	int* vres, bool* bres) {
	TestCore(q, vres, bres);
}

void __device__ DeviceTestCore(size_t cap, int* vres, bool* bres) {
	auto q = cpm::priority_queue<int, PQComparator>(cap);
	TestCore(&q, vres, bres);
	q.free_device();
}
void __global__ TestFromDevice(size_t cap, int* vres, bool* bres) {
	DeviceTestCore(cap, vres, bres);
}

void ctest::PQTest() {
	println_divider();
	std::cout << "Priority Queue Test" << std::endl;
	const size_t res_size = 5;

	int vres_host[res_size];
	int vexpected[] = { 1, 1, -10, -10, -2 };
	bool vmatched[res_size];

	bool bres_host[res_size];
	bool bexpected[] = { true, false, true, false, true };
	bool bmatched[res_size];

	int* vres_device;
	bool* bres_device;
	cudaMalloc((void**)&vres_device, res_size * sizeof(int));
	cudaMalloc((void**)&bres_device, res_size * sizeof(bool));

	size_t c_cap = 10;
	thrust::device_vector<int> dev_vec(c_cap);

	cpm::priority_queue<int, PQComparator>* q;
	cudaMallocManaged((void**)&q, sizeof(cpm::priority_queue<int, PQComparator>));
	new (q) cpm::priority_queue<int, PQComparator>(thrust::raw_pointer_cast(dev_vec.data()), c_cap);

	TestFromHostToDevice<<<1, 1>>>(q, vres_device, bres_device);

	auto cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "Ошибка синхронизации устройства: %s\n", cudaGetErrorString(cudaStatus));
		exit(EXIT_FAILURE);
	}

	std::cout << "Array: ";
	for (int i = 0; i < c_cap; i++) {
		std::cout << dev_vec[i] << " ";
	}
	std::cout << std::endl;

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

	TestFromDevice << <1, 1 >> > (c_cap, vres_device, bres_device);
	cudaStatus = cudaDeviceSynchronize();
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