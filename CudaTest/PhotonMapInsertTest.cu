#include "Test.cuh"
#include "Photon.cuh"
#include "Tree.cuh"
#include "TestablePhotonMap.cuh"

void __host__ __device__ TestCore(cpm::Photon* photons, size_t size, cpm::vec3* vres) {
	auto pm = cpm::TestablePhotonMap(cpm::TestablePhotonMap::caustic, photons, size);

	vres[0] = pm.get_root().pos;
	vres[1] = pm.get_left(0).pos;
	vres[2] = pm.get_left(1).pos;
	vres[3] = pm.get_right(1).pos;
	vres[4] = pm.get_right(0).pos;
	vres[5] = pm.get_left(2).pos;
}

void __global__ TestFromDevice(cpm::Photon* photons, size_t size, cpm::vec3* vres) {
	TestCore(photons, size, vres);
}

void ctest::PhotonMapInsertTest() {
	println_divider();
	std::cout << "Photon Map Insert Test" << std::endl;

	cpm::vec3 vec3_stub = cpm::vec3(1);
	size_t photons_size = 6;
	cpm::Photon photons_data[] = {
		cpm::Photon(cpm::vec3(-10, 7, 20), vec3_stub, vec3_stub),
		cpm::Photon(cpm::vec3(-50, 4, 30), vec3_stub, vec3_stub),
		cpm::Photon(cpm::vec3(0, 3, 21), vec3_stub, vec3_stub),
		cpm::Photon(cpm::vec3(80, 9, 40), vec3_stub, vec3_stub),
		cpm::Photon(cpm::vec3(40, 1, 10), vec3_stub, vec3_stub),
		cpm::Photon(cpm::vec3(30, 5, 0), vec3_stub, vec3_stub)
	};

	/* p - plane
	* -50 -10 0 30 40 80 - first, x
	* 20, 21, 30 - second left, z	; 10, 40 - second right, z
	* 
	*							p=0, (30, 5, 0)
	*						/					\
	*			p=2, (-10, 7, 20)				p=2, (80, 9, 40)
	*			/			\					/
	*		(-50, 4, 30)	(0, 3, 21)	(40, 1, 10)
	* 
	*/

	const size_t res_size = 6;
	cpm::vec3 vres_host[res_size];
	cpm::vec3 vexpected[] = {
		cpm::vec3(30, 5, 0),
		cpm::vec3(-10, 7, 20),
		cpm::vec3(-50, 4, 30),
		cpm::vec3(0, 3, 21),
		cpm::vec3(80, 9, 40),
		cpm::vec3(40, 1, 10),
	};
	bool vmatched[res_size];

	cpm::vec3* vres_device;
	cudaMalloc((void**)&vres_device, res_size * sizeof(cpm::vec3));

	cpm::Photon* photons_device_data;
	cudaMalloc((void**)&photons_device_data, photons_size * sizeof(cpm::Photon));
	cudaMemcpy(photons_device_data, photons_data, photons_size * sizeof(cpm::Photon), cudaMemcpyHostToDevice);
	//---------------------------------------------//
	TestFromDevice << <1, 1 >> > (photons_device_data, photons_size, vres_device);
	check_errors();

	cudaMemcpy(vres_host, vres_device, res_size * sizeof(cpm::vec3), cudaMemcpyDeviceToHost);
	
	is_matched(res_size, vexpected, vres_host, vmatched);
	

	println_results(res_size, vexpected, vres_host, vmatched, "FromDevice");
	//---------------------------------------------//

	cudaFree(vres_device);
	println_divider();
}