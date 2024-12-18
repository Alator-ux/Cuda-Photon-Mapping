#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

template <typename T>
struct Tuple3 {
	T item1, item2, item3;
	__host__ __device__ Tuple3(T item1, T item2, T item3) : item1(item1), item2(item2), item3(item3) {}
};