#pragma once
#include "CudaUtils.cuh"

#define TWO_D_THREADS_NUMBER 16

#define THREADS_NUMBER TWO_D_THREADS_NUMBER * TWO_D_THREADS_NUMBER

#define uint unsigned int

#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )