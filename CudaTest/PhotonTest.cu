#include "Test.cuh"
#include "Photon.cuh"

void __host__ __device__ TestCore(cpm::Photon *photon, float* vres) {
	vres[0] = photon->pos.x();
	vres[1] = photon->pos.y();
	vres[2] = photon->pos.z();

	vres[3] = photon->power.r();
	vres[4] = photon->power.g();
	vres[5] = photon->power.b();

	vres[6] = photon->inc_dir[0];
	vres[7] = photon->inc_dir[1];
	vres[8] = photon->inc_dir[2];
}
void __global__ TestFromHostToDevice(cpm::Photon *photon, float* vres) {
	TestCore(photon, vres);
}

void __device__ DeviceTestCore(float* vres) {
	auto photon = cpm::Photon(vec3(1, 1, 1), vec3(2.0, 1.0, 1.0), vec3(3.f, 1.f, 2.f));
	TestCore(&photon, vres);
}
void __global__ TestFromDevice(float* vres) {
	DeviceTestCore(vres);
}

void __device__ SecondDeviceTestCore(float* vres) {
	auto other_photon = cpm::Photon(vec3(1, 1, 1), vec3(2.0, 1.0, 1.0), vec3(3.f, 1.f, 2.f));
	auto photon = cpm::Photon(other_photon);
	TestCore(&photon, vres);
}
void __global__ SecondTestFromDevice(float* vres) {
	SecondDeviceTestCore(vres);
}

void ctest::PhotonTest() {
	println_divider();
	std::cout << "Photon Test" << std::endl;
	const size_t res_size = 9;

	float vres_host[res_size];
	float vexpected[] = {
		1, 1, 1,
		2, 1, 1,
		3, 1, 2 
	};
	bool vmatched[res_size];

	float* vres_device;
	cudaMalloc((void**)&vres_device, res_size * sizeof(float));

	//---------------------------------------------//
	cpm::Photon* photon;
	cudaMallocManaged((void**)&photon, sizeof(cpm::Photon));
	new (photon) cpm::Photon(vec3(1.f, 1.f, 1.f), vec3(2.f, 1.f, 1.f), vec3(3.f, 1.f, 2.f));

	TestFromHostToDevice << <1, 1 >> > (photon, vres_device);
	check_errors();

	cudaMemcpy(vres_host, vres_device, res_size * sizeof(float), cudaMemcpyDeviceToHost);

	is_matched(res_size, vexpected, vres_host, vmatched);

	println_results(res_size, vexpected, vres_host, vmatched, "FromHostToDevice Part");
	//---------------------------------------------//
	TestFromDevice << <1, 1 >> > (vres_device);
	check_errors();

	cudaMemcpy(vres_host, vres_device, res_size * sizeof(float), cudaMemcpyDeviceToHost);

	is_matched(res_size, vexpected, vres_host, vmatched);

	println_results(res_size, vexpected, vres_host, vmatched, "FromDevice Part");
	//---------------------------------------------//
	SecondTestFromDevice<<<1, 1 >>>(vres_device);
	check_errors();

	cudaMemcpy(vres_host, vres_device, res_size * sizeof(float), cudaMemcpyDeviceToHost);

	is_matched(res_size, vexpected, vres_host, vmatched);

	println_results(res_size, vexpected, vres_host, vmatched, "FromDevice With Copy Other Photon Part");
	//---------------------------------------------//

	cudaFree(vres_device);
	println_divider();
}