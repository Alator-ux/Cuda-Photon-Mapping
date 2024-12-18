#include "Test.cuh"

void ctest::check_errors() {
	auto cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "Ошибка синхронизации устройства: %s\n", cudaGetErrorString(cudaStatus));
		exit(EXIT_FAILURE);
	}
}

void check_cuda(cudaError_t result, char const* const func, const char* const file, int const line) {
	if (result) {
		std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
			file << ":" << line << " '" << func << "' \n";
		// Make sure we call CUDA Device Reset before exiting
		cudaDeviceReset();
		exit(99);
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
