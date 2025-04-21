#pragma once
#include "CudaUtils.cuh"

/* RENDER */
#define TWO_D_THREADS_NUMBER 16
#define THREADS_NUMBER TWO_D_THREADS_NUMBER * TWO_D_THREADS_NUMBER

/* KD TREE */
#define KD_TREE_CHUNK_SIZE 32
#define KD_TREE_EMPTY_SPACE_PERCENTAGE 0.25f

/* AABB */
#define AABB_BITS_TO_ENCODE_IN 4

/* TYPES */
#define uint unsigned int

//#define USE_SIZE_T_AS_IDX

#ifdef USE_SIZE_T_AS_IDX
#define idxtype size_t
#define IDXTYPE_NONE_VALUE SIZE_MAX
#else
#define idxtype uint
#define IDXTYPE_NONE_VALUE UINT_MAX
#endif

/* CUDA UTIL */
#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )

/* List */
// For wild magic
#ifndef __CUDA_ARCH__
namespace cooperative_groups {
	void* this_grid();// { return nullptr; }
}
#endif
#ifdef __CUDA_ARCH__
#define LIST_CUDA_OUTER_PARAMS_DEF \
    cooperative_groups::grid_group grid, bool should_execute = true
#else
#define LIST_CUDA_OUTER_PARAMS_DEF  \
    void* placeholder_param_f = nullptr, bool placeholder_param_s = true /* will be removed by the compiler */
#endif

#ifdef __CUDA_ARCH__
#define LIST_CUDA_INNER_PARAMS_DEF \
    cooperative_groups::grid_group grid
#else
#define LIST_CUDA_INNER_PARAMS_DEF 
#endif

//#define CUDA_ONLY_PARAMS_DEF \
//    CUDA_ONLY_PARAMS_DEF_IMPL()
//
//#ifdef __CUDA_ARCH__
//#define CUDA_ONLY_PARAMS_DEF_IMPL() cooperative_groups::grid_group grid, bool should_execute = true
//#else
//#define CUDA_ONLY_PARAMS_DEF_IMPL() void* placeholder_param = nullptr /*will be removed by compiler*/
//#endif
//#ifdef __CUDA_ARCH__
//#define CUDA_ONLY_PARAMS_DEF cooperative_groups::grid_group grid, bool should_execute = true
//#else
//#define CUDA_ONLY_PARAMS_DEF void* placeholder_param = nullptr /*will be removed by compiler*/
//#endif

/* PRESCAN THREADS */
#define PRESCAN_THREADS 512
#define PRESCAN_BLOCKS  44
#define PRESCAN_BLOCKS_NEXT_POWER_OF_TWO PRESCAN_BLOCKS <= 2  ? 2  : \
										 PRESCAN_BLOCKS <= 4  ? 4  : \
										 PRESCAN_BLOCKS <= 8  ? 8  : \
										 PRESCAN_BLOCKS <= 16 ? 16 : \
										 PRESCAN_BLOCKS <= 32 ? 32 : 64
										
