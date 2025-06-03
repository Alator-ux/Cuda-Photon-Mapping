#pragma once
#include <crt/host_defines.h>
#include <cuda_runtime.h>
#include "Ray.cuh"
#include "MMInnerStack.cuh"
#include "Defines.cuh"
#include "Printer.cuh"
#include "GlobalParams.cuh"


#define MM_OUTER_STACK_OFFSET(array_index, array_capacity, struct_size) ((array_capacity) * (struct_size) + (array_index))

namespace cpm {
    class mm_outer_stack {
    protected:
        idxtype size;

    public:
        __host__ __device__ mm_outer_stack() : size(0) {}

        __host__ __device__ bool isEmpty() const {
            return size == 0;
        }

        __host__ __device__ bool isFull(idxtype capacity) const {
            return capacity == 0 || size > capacity - 1;
        }
        __host__ __device__ void initialize() {
            size = 0;
        }

        __host__ __device__ void push(idxtype array_index, idxtype array_capacity, MMInnerContainer value) {
            if (!isFull(mm_outer_capacity())) {
                idxtype offset = MM_OUTER_STACK_OFFSET(array_index, array_capacity, size);
                mm_outer_data()[offset] = value;
                size += 1;
            }
            else {
                Printer::stack_error("stack is full, but tried to push");
            }
        }
        __host__ __device__ void push_copy(idxtype outer_stack_idx, idxtype other_outer_stack_idx, idxtype array_index, idxtype array_capacity, MMInnerContainer value, idxtype count_from_top = 1, idxtype offset_from_top = 0) {
            auto capacity = mm_outer_capacity();
            if (!isFull(capacity)) {
                idxtype offset = MM_OUTER_STACK_OFFSET(array_index, array_capacity, size);
                mm_outer_data()[offset].copy(value.get_size(), outer_stack_idx, other_outer_stack_idx, array_index, capacity, count_from_top, offset_from_top);
                size++;
            }
            else {
                Printer::stack_error("stack is full, but tried to push");
            }
        }
        __host__ __device__ MMInnerContainer pop(idxtype array_index, idxtype array_capacity) {
            if (size > 0) {
                idxtype offset = MM_OUTER_STACK_OFFSET(array_index, array_capacity, size - 1);
                size -= 1;
                return mm_outer_data()[offset];
            }
        }

        __host__ __device__ MMInnerContainer* top_pointer(idxtype array_index, idxtype array_capacity, idxtype offset = 0) const {
            if (isEmpty()) {
                return nullptr;
            }
            if (size < offset + 1) // size - 1 - offset < 0
            {
                Printer::stack_error("offset more then size, but tried to peek");
                return nullptr;
            }

            return mm_outer_data() + MM_OUTER_STACK_OFFSET(array_index, array_capacity, size - 1 - offset);
        }
        __host__ __device__ MMInnerContainer top(idxtype array_index, idxtype array_capacity, idxtype offset = 0) const {
            if (this->isEmpty())
            {
                Printer::stack_error("stack is empty, but tried to peek");
                return MMInnerContainer();
            }
            if (size < offset + 1) // size - 1 - offset < 0
            {
                Printer::stack_error("offset more then size, but tried to peek");
                return MMInnerContainer();
            }

            return mm_outer_data()[MM_OUTER_STACK_OFFSET(array_index, array_capacity, size - 1 - offset)];
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

#define MMOuterContainer cpm::mm_outer_stack
