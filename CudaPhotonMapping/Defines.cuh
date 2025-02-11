#pragma once
#include "CudaUtils.cuh"

#define THREADS_NUMBER 256

#define uint unsigned int

#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )