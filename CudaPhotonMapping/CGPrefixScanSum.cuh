#pragma once
#include <crt/host_defines.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include "Defines.cuh"
#include "CudaGridSynchronizer.cuh";
#include "SharedMemory.cuh"
#include <cooperative_groups.h>
#include "PrescanCommon.cuh"
#include <cooperative_groups/scan.h>

__device__ void cooperative_inclusive_prescan(PrescanHelperStruct<idxtype> pss, idxtype length, idxtype* global_sum_offset);