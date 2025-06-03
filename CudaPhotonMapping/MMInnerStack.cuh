#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "Printer.cuh"
#include "Defines.cuh"
#include "MediumContent.cuh"

//#define MM_INNER_STACK_OFFSET(outer_idx, inner_idx, depth, outer_cap, depth_cap) ( outer_idx * outer_cap * depth_cap + (inner_idx * depth_cap + depth) )
#define MM_INNER_STACK_OFFSET(array_idx, outer_stack_idx, depth, outer_stack_cap, depth_cap) ( (depth_cap) * ((array_idx) * (outer_stack_cap) + (outer_stack_idx)) + (depth) )


namespace cpm {
    class mm_inner_stack {
    protected:
        idxtype size;
    public:
        __host__ __device__ mm_inner_stack() : size(0) {}

        __host__ __device__ bool isEmpty() const {
            return size == 0;
        }

        __host__ __device__ bool isFull(idxtype capacity) const {
            return capacity == 0 || size > capacity - 1;
        }
        
        __host__ __device__ void initialize() {
            size = 0;
        }

        __host__ __device__ void copy(idxtype other_size,
            idxtype outer_stack_idx, idxtype other_outer_stack_idx,
            idxtype array_index, 
            idxtype outer_capacity,
            int count_from_top = 1, int offset_from_top = 0) {

            int to = max(other_size - offset_from_top, 0);
            int from = max(to - count_from_top, 0);

            auto depth_capacity = mm_inner_capacity();
            idxtype base_offset = MM_INNER_STACK_OFFSET(array_index, outer_stack_idx, 0, outer_capacity, depth_capacity);
            idxtype other_base_offset = MM_INNER_STACK_OFFSET(array_index, other_outer_stack_idx, 0, outer_capacity, depth_capacity);
            auto data = mm_inner_data();
            for (idxtype i = from; i < to; i++) {
                data[base_offset + i] = data[other_base_offset + i];
            }
            this->size = to - from;
        }

        __host__ __device__ void push(idxtype array_index, idxtype outer_stack_idx, idxtype outer_capacity, MMInnerData value) {
            auto depth_capacity = mm_inner_capacity();
            if (!isFull(depth_capacity)) {
                idxtype offset = MM_INNER_STACK_OFFSET(array_index, outer_stack_idx, size, outer_capacity, depth_capacity);
                mm_inner_data()[offset] = value;
                size += 1;
            }
            else {
                Printer::stack_error("Inner stack is full, but tried to push");
            }
        }

        __host__ __device__ MMInnerData pop(idxtype array_index, idxtype outer_stack_idx, idxtype outer_capacity) {
            if (size > 0) {
                auto depth_capacity = mm_inner_capacity();
                idxtype offset = MM_INNER_STACK_OFFSET(array_index, outer_stack_idx, size - 1, outer_capacity, depth_capacity);
                size -= 1;
                return mm_inner_data()[offset];
            }
        }

        __host__ __device__ MMInnerData* top_pointer(idxtype array_index, idxtype outer_stack_idx, idxtype outer_capacity, idxtype offset = 0) const {
            if (isEmpty()) {
                return nullptr;
            }
            if (size < offset + 1) // size - 1 - offset < 0
            {
                Printer::stack_error("offset more then size, but tried to peek");
                return nullptr;
            }
            auto depth_capacity = mm_inner_capacity();
            idxtype global_offset = MM_INNER_STACK_OFFSET(array_index, outer_stack_idx, size - 1 - offset, outer_capacity, depth_capacity);
            return mm_inner_data() + global_offset;
        }
        __host__ __device__ MMInnerData top(idxtype array_index, idxtype outer_stack_idx, idxtype outer_capacity, idxtype offset = 0) const {
            if (this->isEmpty())
            {
                Printer::stack_error("stack is empty, but tried to peek");
                return MMInnerData();
            }
            if (size < offset + 1) // size - 1 - offset < 0
            {
                Printer::stack_error("offset more then size, but tried to peek");
                return MMInnerData();
            }

            auto depth_capacity = mm_inner_capacity();
            idxtype global_offset = MM_INNER_STACK_OFFSET(array_index, outer_stack_idx, size - 1 - offset, outer_capacity, depth_capacity);
            return mm_inner_data()[global_offset];
        }
        __host__ __device__ size_t get_size() const {
            return size;
        }
        __host__ __device__ void set_size(int size) {
            if (size >= 0) {
                this->size = size;
            }
        }
    };
}

/* Outer MM Stack */
#define MMInnerContainer cpm::mm_inner_stack

extern MMInnerContainer* cpu_mm_outer_data;
extern __constant__ MMInnerContainer* gpu_mm_outer_data;

__host__ __device__ __forceinline__ MMInnerContainer* mm_outer_data() {
#ifdef __CUDA_ARCH__
    return gpu_mm_outer_data;
#else
    return cpu_mm_outer_data;
#endif
}

extern idxtype cpu_mm_outer_capacity;
extern __constant__ idxtype gpu_mm_outer_capacity;

__host__ __device__ __forceinline__ idxtype mm_outer_capacity() {
#ifdef __CUDA_ARCH__
    return gpu_mm_outer_capacity;
#else
    return cpu_mm_outer_capacity;
#endif
}

__host__ void set_mm_inner_stack_parameters(MMInnerContainer* cpu_mm_outer_data_val, MMInnerContainer* gpu_mm_outer_data_val, idxtype mm_outer_capacity_val);
