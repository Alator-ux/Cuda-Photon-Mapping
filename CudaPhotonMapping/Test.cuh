#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "CudaUtils.cuh"
#include "thrust/host_vector.h"
#include "Stack.cuh"
#include "Printers.h"
#include "PriorityQueue.cuh"
#include "Photon.cuh"
#include "Tree.cuh"
#include "Ray.cuh"



namespace ctest {
	void PQTest();
	void StackTest();
	void PhotonTest();
	void TreeTest();
	void PhotonMapInsertTest();
	void PhotonMapGetClosestTest();
	void RayTracingTest();

	void TestAll();

	template<typename T>
	void is_matched(size_t size, T* expected, T* res, bool* out) {
		for (auto i = 0; i < size; i++) {
			out[i] = expected[i] == res[i];
		}
	}
	void check_errors();

	template<typename T>
	void println_results(size_t size, T* expected, T* real, bool* matched, std::string title) {
		std::cout << "-----" << std::endl;
		std::cout << title << std::endl;
		std::cout << "Expected, result and match values:" << std::endl;
		println_array(expected, size);
		println_array(real, size);
		println_array(matched, size);
		std::cout << "-----" << std::endl;
	}
}