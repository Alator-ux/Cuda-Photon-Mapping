#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "Printer.cuh"
#include "Defines.cuh"

#ifdef __CUDA_ARCH__
//#define STACK_OFFSET(stack_size) ( (threadIdx.x + blockIdx.x * blockDim.x) + stack_size * blockDim.x * gridDim.x )
//#define STACK_SHIFTED_DATA(data, stack_size) data + STACK_OFFSET(stack_size)
#define NESTED_STACK_OFFSET(outer_idx, inner_idx, depth, outer_cap, depth_cap) \
    (((outer_idx) * (outer_cap) + (inner_idx)) * (depth_cap) + (depth))
#define NESTED_STACK_SHIFTED_DATA(data, outer_idx, inner_idx, depth, outer_cap, depth_cap) (data + NESTED_STACK_OFFSET(outer_idx, inner_idx, depth, inner_cap, depth_cap))
#else
#define NESTED_STACK_OFFSET(outer_idx, inner_idx, depth, outer_cap, depth_cap) ( depth )
#define NESTED_STACK_SHIFTED_DATA(data, outer_idx, inner_idx, depth, outer_cap, depth_cap) (data + NESTED_STACK_OFFSET(outer_idx, inner_idx, depth, outer_cap, depth_cap))
#endif


namespace cpm {
    template<typename ElemType>
    class nested_stack {
    protected:
        idxtype size;
    public:
        __host__ __device__ nested_stack(ElemType*& data, idxtype capacity) {
            data = new ElemType[capacity];
            size = 0;
        }

        __host__ __device__ nested_stack() : size(0) {}

        __host__ __device__ bool isEmpty() const {
            return size == 0;
        }

        __host__ __device__ bool isFull(idxtype capacity) const {
            return capacity == 0 || size > capacity - 1;
        }
        __host__ __device__ void initialize(ElemType*& data, idxtype& capacity, idxtype new_capacity) {
            free(data);
            data = (ElemType*)malloc(new_capacity * sizeof(ElemType));
            capacity = new_capacity;
        }
        __host__ __device__ void initialize() {
            size = 0;
        }

        __host__ __device__ void copy(ElemType* data, idxtype other_size, 
            idxtype outer_stack_idx, idxtype other_outer_stack_idx, 
            idxtype array_index, idxtype array_capacity, 
            idxtype outer_capacity, idxtype capacity,
            int count_from_top = 1, int offset_from_top = 0) {
            /*if (capacity != other_capacity) {
                Printer::stack_error("this capacity != other capacity when copy");
                return;
            }*/
            int to = max((int)other_size - offset_from_top, 0);
            int from = max(to - count_from_top, 0);
            idxtype base_offset = NESTED_STACK_OFFSET(array_index, outer_stack_idx, 0, array_capacity, capacity);
            idxtype other_base_offset = NESTED_STACK_OFFSET(array_index, other_outer_stack_idx, 0, array_capacity, capacity);
            for (int i = from; i < to; i++) {
                data[base_offset + i] = data[other_base_offset + i];
            }
            this->size = to - from;
        }

        __host__ __device__ void push(ElemType* data, idxtype array_index, idxtype outer_stack_idx, idxtype array_capacity, idxtype capacity, ElemType value) {
            if (!isFull(capacity)) {
                idxtype offset = NESTED_STACK_OFFSET(array_index, outer_stack_idx, size, array_capacity, capacity);
                data[offset] = value;
                size += 1;
            }
            else {
                Printer::stack_error("stack is full, but tried to push");
            }
        }

        __host__ __device__ ElemType pop(ElemType* data, idxtype array_index, idxtype outer_stack_idx, idxtype array_capacity, idxtype capacity) {
            if (size > 0) {
                idxtype offset = NESTED_STACK_OFFSET(array_index, outer_stack_idx, size, array_capacity, capacity);
                size -= 1;
                return data[offset];
            }
        }

        __host__ __device__ ElemType* top_pointer(ElemType* data, idxtype array_index, idxtype outer_stack_idx, idxtype array_capacity, idxtype capacity, int offset = 0) const {
            if (isEmpty()) {
                return nullptr;
            }
            if (size - 1 - offset < 0)
            {
                Printer::stack_error("offset more then size, but tried to peek");
                return nullptr;
            }
            idxtype global_offset = NESTED_STACK_OFFSET(array_index, outer_stack_idx, size - 1 - offset, array_capacity, capacity);
            return data + global_offset;
        }
        __host__ __device__ ElemType top(ElemType* data, idxtype array_index, idxtype outer_stack_idx, idxtype array_capacity, idxtype capacity, int offset = 0) const {
            if (this->isEmpty())
            {
                Printer::stack_error("stack is empty, but tried to peek");
                return ElemType();
            }
            if (size - 1 - offset < 0)
            {
                Printer::stack_error("offset more then size, but tried to peek");
                return ElemType();
            }

            idxtype global_offset = NESTED_STACK_OFFSET(array_index, outer_stack_idx, size - 1 - offset, array_capacity, capacity);
            return data[global_offset];
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