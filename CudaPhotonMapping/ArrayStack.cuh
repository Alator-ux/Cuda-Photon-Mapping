#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "Printer.cuh"
#include "Defines.cuh"

// stack index in array, array cap/size, elem index in current stack
#define ARRAY_STRUCT_OFFSET(array_index, array_capacity, struct_size) (array_capacity * struct_size + array_index)
#define ARRAY_STRUCT_SHIFTED_DATA(data, array_index, array_capacity, struct_size) (data + ARRAY_STRUCT_OFFSET(array_index, array_capacity, struct_size))

namespace cpm {
    template<typename ElemType>
    class array_stack {
    public:
        struct stack_content {
            idxtype size;
            idxtype capacity;
            ElemType* data;
        };
        struct stack_outer_content {
            idxtype capacity;
            ElemType* data;
        };
    protected:
        idxtype size;
        
    public:
        __host__ __device__ array_stack() : size(0) {}

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
        /*
        ElemType* data, idxtype other_size, 
            idxtype outer_stack_idx, idxtype other_outer_stack_idx, 
            idxtype array_index, idxtype array_capacity, 
            idxtype outer_capacity, idxtype capacity,
            int count_from_top = 1, int offset_from_top = 0
            */
        template<typename inner_data_type>
        __host__ __device__ void push_copy(ElemType* data, inner_data_type* inner_data, idxtype capacity, idxtype inner_capacity, idxtype outer_stack_idx, idxtype other_outer_stack_idx, idxtype array_index, idxtype array_capacity, ElemType value, int count_from_top = 1, int offset_from_top = 0) {
            if (!isFull(capacity)) {
                idxtype offset = ARRAY_STRUCT_OFFSET(array_index, array_capacity, size);
                data[offset].copy(inner_data, value.get_size(), outer_stack_idx, other_outer_stack_idx, array_index, array_capacity, capacity, inner_capacity, count_from_top, offset_from_top);
                size++;
            }
            else {
                Printer::stack_error("stack is full, but tried to push");
            }
        }
        __host__ __device__ void push(ElemType* data, idxtype capacity, idxtype array_index, idxtype array_capacity, ElemType value) {
            if (!isFull(capacity)) {
                idxtype offset = ARRAY_STRUCT_OFFSET(array_index, array_capacity, size);
                data[offset] = value;
                size += 1;
            }
            else {
                Printer::stack_error("stack is full, but tried to push");
            }
        }
        __host__ __device__ ElemType pop(ElemType* data, idxtype array_index, idxtype array_capacity) {
            if (size > 0) {
                idxtype offset = ARRAY_STRUCT_OFFSET(array_index, array_capacity, size);
                size -= 1;
                return data[offset];
            }
        }

        __host__ __device__ ElemType* top_pointer(ElemType* data, idxtype array_index, idxtype array_capacity, int offset = 0) const {
            if (isEmpty()) {
                return nullptr;
            }
            if (size - 1 - offset < 0)
            {
                Printer::stack_error("offset more then size, but tried to peek");
                return nullptr;
            }

            return ARRAY_STRUCT_SHIFTED_DATA(data, array_index, array_capacity, size - 1 - offset);
        }
        __host__ __device__ ElemType top(ElemType* data, idxtype array_index, idxtype array_capacity, int offset = 0) const {
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

            return *ARRAY_STRUCT_SHIFTED_DATA(data, array_index, array_capacity, size - 1 - offset);
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