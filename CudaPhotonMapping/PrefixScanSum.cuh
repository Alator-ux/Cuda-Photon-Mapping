#pragma once
#include <crt/host_defines.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include "Defines.cuh"
#include "CudaGridSynchronizer.cuh";
#include "SharedMemory.cuh"
#include <cooperative_groups.h>
#include "PrescanCommon.cuh"
#include "MathFunctions.cuh"

template <typename T, bool inclusive = true, bool need_total_sum = false>
__device__ void prescan(PrescanHelperStruct<T> pss, idxtype length, T* global_sum_offset) {
    constexpr int elements_per_block = (PRESCAN_THREADS * 2);
    constexpr idxtype elements_per_iteration = elements_per_block * PRESCAN_BLOCKS;
    const idxtype total_iterations = length / elements_per_iteration;
    T sum_offset = 0;

    if (total_iterations > 0) {
        pss.out_arr += elements_per_block * blockIdx.x;
        pss.in_arr += elements_per_block * blockIdx.x;
    }
    for (idxtype i = 0; i < total_iterations; i++) {
        prescan_full_threads(pss.out_arr, pss.in_arr, pss.separated_sums_arr, elements_per_block, inclusive);
        CudaGridSynchronizer::synchronize_grid();
        prescan_arbitrary(pss.united_sums_arr, pss.separated_sums_arr, PRESCAN_BLOCKS, PRESCAN_BLOCKS_NEXT_POWER_OF_TWO);
        CudaGridSynchronizer::synchronize_grid();
        if (blockIdx.x == PRESCAN_BLOCKS - 1) {
            T shift = pss.united_sums_arr[blockIdx.x];
            
            shift += sum_offset;
            idxtype out_arr_idx = threadIdx.x;
            pss.out_arr[out_arr_idx] += shift;
            pss.out_arr[out_arr_idx + PRESCAN_THREADS] += shift;

            if (threadIdx.x == PRESCAN_THREADS - 1) {
                idxtype last_idx = PRESCAN_THREADS * 2 - 1; // threadIdx.x + PRESCAN_THREADS
                T offset_to_add = pss.out_arr[last_idx] - sum_offset;
                if (!inclusive) {
                    offset_to_add += pss.in_arr[last_idx];
                }
                *global_sum_offset += offset_to_add;
            }
        }
        else if (blockIdx.x > 0) {
            
            idxtype out_arr_idx = threadIdx.x;
            T shift = pss.united_sums_arr[blockIdx.x] + sum_offset;
            pss.out_arr[out_arr_idx] += shift;
            pss.out_arr[out_arr_idx + PRESCAN_THREADS] += shift;
        }
        else { // first block
            idxtype out_arr_idx = threadIdx.x;
            T shift = sum_offset;
            pss.out_arr[out_arr_idx] += shift;
            pss.out_arr[out_arr_idx + PRESCAN_THREADS] += shift;

            /*T shift = T();
            for (idxtype sum_idx = threadIdx.x; sum_idx < gridDim.x; sum_idx += blockDim.x) {
                shift += sums_arr[sum_idx];
            }
            shift = warp_reduce_sum(shift);
            if (threadIdx.x % 32 == 0) {
                atomicAdd(global_shift, shift);
            }*/
        }
        pss.out_arr += elements_per_iteration;
        pss.in_arr += elements_per_iteration;
        CudaGridSynchronizer::synchronize_grid();
        sum_offset = *global_sum_offset;
    }

    idxtype leftover = length - total_iterations * elements_per_iteration;
    if (leftover == 0) return;

    idxtype last_block_idx = leftover / elements_per_block;
    leftover = leftover - last_block_idx * elements_per_block;
    if (blockIdx.x < last_block_idx) {
        if (total_iterations == 0) {
            pss.out_arr += elements_per_block * blockIdx.x;
            pss.in_arr += elements_per_block * blockIdx.x;
        }
        prescan_full_threads(pss.out_arr, pss.in_arr, pss.separated_sums_arr, elements_per_block, inclusive);
    }
    else if (leftover > 0 && blockIdx.x == last_block_idx) {
        if (total_iterations == 0) {
            pss.out_arr += elements_per_block * last_block_idx;
            pss.in_arr += elements_per_block * last_block_idx;
        }
        prescan_arbitrary(pss.out_arr, pss.in_arr, leftover, next_power_of_two(leftover));
    }
    
    bool hybrid_case = leftover > 0 && last_block_idx > 0;
   
    CudaGridSynchronizer::synchronize_grid();
    if (blockIdx.x < last_block_idx && blockIdx.x == 0) {
        prescan_arbitrary(pss.united_sums_arr, pss.separated_sums_arr, 
            last_block_idx + (hybrid_case ? 1 : 0), next_power_of_two(last_block_idx + (hybrid_case ? 1 : 0)));
    }
    CudaGridSynchronizer::synchronize_grid();
    
    if (blockIdx.x < last_block_idx) {
        idxtype out_arr_idx = threadIdx.x;
        T shift = pss.united_sums_arr[blockIdx.x] + sum_offset;
        pss.out_arr[out_arr_idx] += shift;
        out_arr_idx += PRESCAN_THREADS;
        pss.out_arr[out_arr_idx] += shift;
        if (!inclusive && need_total_sum && out_arr_idx == length - 1) {
            *global_sum_offset = pss.out_arr[out_arr_idx] + pss.in_arr[out_arr_idx];
        }
    }
    else if(leftover > 0 && blockIdx.x == last_block_idx) {
        idxtype out_arr_idx = threadIdx.x;
        T shift = sum_offset;
        if (hybrid_case) {
            shift += pss.united_sums_arr[blockIdx.x];
        }
        
        if (out_arr_idx < leftover) {
            if (inclusive) {
                pss.out_arr[out_arr_idx] += shift + pss.in_arr[out_arr_idx];
            }
            else {
                pss.out_arr[out_arr_idx] += shift;
                if (need_total_sum) {
                    if (out_arr_idx == length - 1) {
                        *global_sum_offset = pss.out_arr[out_arr_idx] + pss.in_arr[out_arr_idx];
                    }
                }
            }
        }
        out_arr_idx += PRESCAN_THREADS;
        if (out_arr_idx < leftover) {
            if (inclusive) {
                pss.out_arr[out_arr_idx] += shift + pss.in_arr[out_arr_idx];
            }
            else {
                pss.out_arr[out_arr_idx] += shift;
                if (need_total_sum) {
                    if (out_arr_idx == length - 1) {
                        *global_sum_offset = pss.out_arr[out_arr_idx] + pss.in_arr[out_arr_idx];
                    }
                }
            }
        }

    }
}

template <typename T>
__device__ void prescan_full_threads(T* out_arr, T* in_data, T* sums, idxtype n, bool inclusive) {
    SharedMem<T> shared;
    T* smem = shared.getPointer();

    int thid = threadIdx.x;


    int ai = thid;
    int bi = thid + (n / 2);
    idxtype bankOffsetA = CONFLICT_FREE_OFFSET(ai);
    idxtype bankOffsetB = CONFLICT_FREE_OFFSET(bi);
    T in_data_ai = in_data[ai];
    T in_data_bi = in_data[bi];
    smem[ai + bankOffsetA] = in_data_ai;
    smem[bi + bankOffsetB] = in_data_bi;

    idxtype offset = 1;
    for (idxtype d = n >> 1; d > 0; d = d >> 1) {
        __syncthreads();
        if (thid < d) {
            idxtype ai = offset * (2 * thid + 1) - 1;
            idxtype bi = offset * (2 * thid + 2) - 1;
            ai += CONFLICT_FREE_OFFSET(ai);
            bi += CONFLICT_FREE_OFFSET(bi);

            smem[bi] += smem[ai];
        }
        offset *= 2;
    }

    if (thid == 0)
    {
        sums[blockIdx.x] = smem[n - 1 + CONFLICT_FREE_OFFSET(n - 1)];
        smem[n - 1 + CONFLICT_FREE_OFFSET(n - 1)] = 0;
    } // clear the last element

    for (int d = 1; d < n; d *= 2) // traverse down tree & build scan
    {
        offset >>= 1;
        __syncthreads();
        if (thid < d) {
            int ai = offset * (2 * thid + 1) - 1;
            int bi = offset * (2 * thid + 2) - 1;
            ai += CONFLICT_FREE_OFFSET(ai);
            bi += CONFLICT_FREE_OFFSET(bi);

            float t = smem[ai];
            smem[ai] = smem[bi];
            smem[bi] += t;
        }
    }
    __syncthreads();
    
    if (inclusive) {
        out_arr[ai] = smem[ai + bankOffsetA] + in_data_ai; // write results to device memory
        out_arr[bi] = smem[bi + bankOffsetB] + in_data_bi;
    }
    else {
        out_arr[ai] = smem[ai + bankOffsetA]; // write results to device memory
        out_arr[bi] = smem[bi + bankOffsetB];
    }
}



template <typename T>
__device__ void prescan_arbitrary(T* out_arr, T* in_arr, idxtype length, int powerOfTwo) {
    SharedMem<T> shared;
    T* smem = shared.getPointer(); // allocated on invocation
    int threadID = threadIdx.x;

    int ai = threadID;
    int bi = threadID + (length / 2);
    int bankOffsetA = CONFLICT_FREE_OFFSET(ai);
    int bankOffsetB = CONFLICT_FREE_OFFSET(bi);

    if (threadID < length) {
        smem[ai + bankOffsetA] = in_arr[ai];
        smem[bi + bankOffsetB] = in_arr[bi];
    }
    else {
        smem[ai + bankOffsetA] = 0;
        smem[bi + bankOffsetB] = 0;
    }


    int offset = 1;
    for (int d = powerOfTwo >> 1; d > 0; d >>= 1) // build sum in place up the tree
    {
        __syncthreads();
        if (threadID < d)
        {
            int ai = offset * (2 * threadID + 1) - 1;
            int bi = offset * (2 * threadID + 2) - 1;
            ai += CONFLICT_FREE_OFFSET(ai);
            bi += CONFLICT_FREE_OFFSET(bi);

            smem[bi] += smem[ai];
        }
        offset *= 2;
    }

    if (threadID == 0) {
        smem[powerOfTwo - 1 + CONFLICT_FREE_OFFSET(powerOfTwo - 1)] = 0; // clear the last element
    }

    for (int d = 1; d < powerOfTwo; d *= 2) // traverse down tree & build scan
    {
        offset >>= 1;
        __syncthreads();
        if (threadID < d)
        {
            int ai = offset * (2 * threadID + 1) - 1;
            int bi = offset * (2 * threadID + 2) - 1;
            ai += CONFLICT_FREE_OFFSET(ai);
            bi += CONFLICT_FREE_OFFSET(bi);

            int t = smem[ai];
            smem[ai] = smem[bi];
            smem[bi] += t;
        }
    }
    __syncthreads();

    if (threadID < length) {
        out_arr[ai] = smem[ai + bankOffsetA];
        out_arr[bi] = smem[bi + bankOffsetB];
    }
}


