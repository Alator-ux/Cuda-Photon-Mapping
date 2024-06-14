#include "Test.cuh"
#include "Photon.cuh"
#include "Tree.cuh"

void __host__ __device__ TestCore(cpm::Tree<cpm::Photon>* tree, float* vres, bool* bres) {
	auto root = cpm::Photon(cpm::vec3(1), cpm::vec3(1), cpm::vec3(1));
	auto left = cpm::Photon(cpm::vec3(2), cpm::vec3(2), cpm::vec3(2));
	auto right = cpm::Photon(cpm::vec3(3), cpm::vec3(3), cpm::vec3(3));
	auto left_right = cpm::Photon(cpm::vec3(4), cpm::vec3(4), cpm::vec3(4));
	tree->set_root(&root);
	int left_ind = tree->add_left(0, &left);
	int right_ind = tree->add_right(0, &right);
	int left_right_ind = tree->add_right(left_ind, &left_right);

	bres[0] = tree->has_root();
	bres[1] = tree->has_left(0) == true;
	bres[2] = tree->has_left(left_ind) == false;
	bres[3] = tree->has_right(0) == true;
	bres[4] = tree->has_right(left_ind) == true;
	bres[5] = tree->has_right(right_ind) == false;
	bres[6] = tree->has_left(10) == false;

	vres[0] = tree->get_root()->pos.x();
	vres[1] = tree->get_right(0)->pos.x();
	vres[2] = tree->get_right(right_ind) == nullptr ? -1 : tree->get_right(right_ind)->pos.x();
	vres[3] = tree->get_left(0)->pos.x();
	vres[4] = tree->get_right(left_ind)->pos.x();
	vres[5] = tree->add_right(10, &right);
	vres[6] = tree->get_left(10) == nullptr ? -1 : tree->get_left(10)->pos.x();
}
void __global__ TestFromHostToDevice(cpm::Tree<cpm::Photon>* tree, float* vres, bool* bres) {
	TestCore(tree, vres, bres);
}

void __device__ DeviceTestCore(float* vres, bool* bres) {
	auto tree = cpm::Tree<cpm::Photon>(8);
	TestCore(&tree, vres, bres);
}
void __global__ TestFromDevice(float* vres, bool* bres) {
	DeviceTestCore(vres, bres);
}

void ctest::TreeTest() {
	println_divider();
	std::cout << "Photon Test" << std::endl;
	const size_t res_size = 7;

	float vres_host[res_size];
	float vexpected[] = {
		1, 3, -1, 2, 4, -1, -1
	};
	bool vmatched[res_size];

	bool bres_host[res_size];
	bool bexpected[] = {
		true, true, true, 
		true, true, true,
		true
	};
	bool bmatched[res_size];

	float* vres_device;
	cudaMalloc((void**)&vres_device, res_size * sizeof(float));
	bool* bres_device;
	cudaMalloc((void**)&bres_device, res_size * sizeof(bool));

	//---------------------------------------------//
	/*int cap = 8;
	cpm::Tree<cpm::Photon>* tree;
	cudaMallocManaged((void**)&tree, sizeof(cpm::Tree<cpm::Photon>));
	new (tree) cpm::Tree<cpm::Photon>(cap);

	TestFromHostToDevice << <1, 1 >> > (tree, vres_device, bres_device);
	check_errors();

	cudaMemcpy(vres_host, vres_device, res_size * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(bres_host, bres_device, res_size * sizeof(bool), cudaMemcpyDeviceToHost);

	is_matched(res_size, vexpected, vres_host, vmatched);
	is_matched(res_size, bexpected, bres_host, bmatched);

	println_results(res_size, vexpected, vres_host, vmatched, "FromHostToDevice Part 1");
	println_results(res_size, bexpected, bres_host, bmatched, "FromHostToDevice Part 2");*/
	//---------------------------------------------//
	TestFromDevice << <1, 1 >> > (vres_device, bres_device);
	check_errors();

	cudaMemcpy(vres_host, vres_device, res_size * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(bres_host, bres_device, res_size * sizeof(bool), cudaMemcpyDeviceToHost);

	is_matched(res_size, vexpected, vres_host, vmatched);
	is_matched(res_size, bexpected, bres_host, bmatched);

	println_results(res_size, vexpected, vres_host, vmatched, "FromDevice Part 1");
	println_results(res_size, bexpected, bres_host, bmatched, "FromDevice Part 2");
	//---------------------------------------------//

	cudaFree(vres_device);
	println_divider();
}