#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

namespace cpm {
	template <typename T1, typename T2, typename T3>
	struct Tuple3 {
		T1 item1;
		T2 item2;
		T3 item3;
		__host__ __device__ Tuple3(T1 item1, T2 item2, T3 item3) : item1(item1), item2(item2), item3(item3) {}
	};
};