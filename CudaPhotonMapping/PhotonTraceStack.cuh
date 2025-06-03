#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "Printer.cuh"
#include "Defines.cuh"
#include "MediumContent.cuh"




extern MMInnerData* cpu_pt_stack_data;
extern __constant__ MMInnerData* gpu_pt_stack_data;

__host__ __device__ __forceinline__ MMInnerData* pt_stack_data() {
#ifdef __CUDA_ARCH__
    return gpu_pt_stack_data;
#else
    return cpu_pt_stack_data;
#endif
}

extern idxtype cpu_pt_stack_capacity;
extern __constant__ idxtype gpu_pt_stack_capacity;

__host__ __device__ __forceinline__ idxtype pt_stack_capacity() {
#ifdef __CUDA_ARCH__
    return gpu_pt_stack_capacity;
#else
    return cpu_pt_stack_capacity;
#endif
}

__host__ void set_pt_stack_parameters(MMInnerData* cpu_pt_stack_data_val,
    MMInnerData* gpu_pt_stack_data_val, idxtype pt_stack_capacity_val);


#define PHOTON_TRACE_STACK_OFFSET(array_idx, array_size, depth) ( (array_size) * (depth) + (array_idx) )

namespace cpm {
    class photon_trace_stack {
    protected:
        idxtype size;
    public:
        __host__ __device__ photon_trace_stack() : size(0) {}

        __host__ __device__ bool isEmpty() const {
            return size == 0;
        }

        __host__ __device__ bool isFull(idxtype capacity) const {
            return capacity == 0 || size > capacity - 1;
        }

        __host__ __device__ void initialize() {
            size = 0;
        }

        __host__ __device__ void push(idxtype array_index, idxtype array_size, MMInnerData value) {
            auto depth_capacity = pt_stack_capacity();
            if (!isFull(depth_capacity)) {
                idxtype offset = PHOTON_TRACE_STACK_OFFSET(array_index, array_size, size);
                pt_stack_data()[offset] = value;
                size += 1;
            }
            else {
                Printer::stack_error("Inner stack is full, but tried to push");
            }
        }

        __host__ __device__ MMInnerData pop(idxtype array_index, idxtype array_size) {
            if (size > 0) {
                auto depth_capacity = pt_stack_capacity();
                size -= 1;
                idxtype offset = PHOTON_TRACE_STACK_OFFSET(array_index, array_size, size);
                return pt_stack_data()[offset];
            }
        }

        __host__ __device__ MMInnerData* top_pointer(idxtype array_index, idxtype array_size, idxtype offset = 0) const {
            if (isEmpty()) {
                return nullptr;
            }
            if (size < offset + 1) // size - 1 - offset < 0
            {
                Printer::stack_error("offset more then size, but tried to peek");
                return nullptr;
            }
            auto depth_capacity = pt_stack_capacity();
            idxtype global_offset = PHOTON_TRACE_STACK_OFFSET(array_index, array_size, size - 1 - offset);
            return pt_stack_data() + global_offset;
        }
        __host__ __device__ MMInnerData top(idxtype array_index, idxtype array_size, idxtype offset = 0) const {
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

            auto depth_capacity = pt_stack_capacity();
            idxtype global_offset = PHOTON_TRACE_STACK_OFFSET(array_index, array_size, size - 1 - offset);
            return pt_stack_data()[global_offset];
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

