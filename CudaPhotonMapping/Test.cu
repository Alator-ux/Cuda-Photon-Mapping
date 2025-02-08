#include "Test.cuh"

void ctest::check_errors() {
	auto cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "Ошибка синхронизации устройства: %s\n", cudaGetErrorString(cudaStatus));
		exit(EXIT_FAILURE);
	}
}

void ctest::TestAll() {
	ctest::PQTest();
	ctest::StackTest();
	ctest::PhotonTest();
	ctest::TreeTest();
	ctest::PhotonMapInsertTest();
	ctest::PhotonMapGetClosestTest();
	ctest::RayTracingTest();
}
