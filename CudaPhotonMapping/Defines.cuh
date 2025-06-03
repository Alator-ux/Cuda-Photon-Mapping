#pragma once
#include "CudaUtils.cuh"
#include <curand_kernel.h>

/* RENDER */
#define TWO_D_THREADS_NUMBER 16
#define THREADS_NUMBER TWO_D_THREADS_NUMBER * TWO_D_THREADS_NUMBER

/* KD TREE */
#define KD_TREE_CHUNK_SIZE 32
#define KD_TREE_EMPTY_SPACE_PERCENTAGE 0.25f

/* AABB */
#define AABB_BITS_TO_ENCODE_IN 4

/* TYPES */
typedef unsigned int uint;

typedef unsigned long long uint64;

typedef curandStatePhilox4_32_10_t cudaRandomStateT;

//#define USE_SIZE_T_AS_IDX

#ifdef USE_SIZE_T_AS_IDX
typedef size_t idxtype;
#define IDXTYPE_NONE_VALUE SIZE_MAX
#define IDX_ONE 1ull
#else
typedef uint idxtype;
#define IDXTYPE_NONE_VALUE UINT_MAX
#define IDX_ONE 1u
#endif

/* CUDA UTIL */
#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )

/* List */
// For wild magic
#ifndef __CUDA_ARCH__
namespace cooperative_groups {
	struct grid_group {
		static grid_group instance;
		void sync();
	};
	grid_group this_grid();// { return nullptr; }
}
#endif
#ifdef __CUDA_ARCH__
#define LIST_CUDA_OUTER_PARAMS_DEF \
    cooperative_groups::grid_group grid, bool should_execute = true
#else
#define LIST_CUDA_OUTER_PARAMS_DEF  \
    cooperative_groups::grid_group placeholder_param_f = cooperative_groups::grid_group::instance, bool placeholder_param_s = true /* will be removed by the compiler */
#endif

#ifdef __CUDA_ARCH__
#define LIST_CUDA_INNER_PARAMS_DEF \
    cooperative_groups::grid_group grid
#else
#define LIST_CUDA_INNER_PARAMS_DEF 
#endif

/* PRESCAN THREADS */
#define PRESCAN_THREADS 512
#define PRESCAN_BLOCKS  44
#define PRESCAN_BLOCKS_NEXT_POWER_OF_TWO PRESCAN_BLOCKS <= 2  ? 2  : \
										 PRESCAN_BLOCKS <= 4  ? 4  : \
										 PRESCAN_BLOCKS <= 8  ? 8  : \
										 PRESCAN_BLOCKS <= 16 ? 16 : \
										 PRESCAN_BLOCKS <= 32 ? 32 : 64

										
