#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "Printer.cuh"

namespace cpm {
    template<typename ElemType>
    class stack {
    protected:
        ElemType* data;
        size_t size, capacity;
        __host__ void resize() {
            size_t new_capacity = capacity == 0 ? 1 : capacity * 2;
            ElemType* new_data = (ElemType*)malloc(new_capacity * sizeof(ElemType));

            for (size_t i = 0; i < capacity; i++) {
                new_data[i] = data[i];
            }

            free((void*)data);
            data = new_data;
            capacity = new_capacity;
        }

    public:
        __host__ __device__ stack(size_t capacity) {
            data = new ElemType[capacity];
            size = 0;
            this->capacity = capacity == 0 ? 0 : capacity - 1;
        }

        __host__ __device__ stack() : data(nullptr), size(0), capacity(0) {}

        // :(
        /*__host__ __device__ ~stack() {
            delete[] data;
        }*/

        __host__ __device__ bool isEmpty() const {
            return size == 0;
        }

        __host__ __device__ bool isFull() const {
            return capacity == 0 || size > capacity;
        }
        __host__ __device__ void initialize(size_t new_capacity) {
            free(data);
            data = (ElemType*)malloc(new_capacity * sizeof(ElemType));
            capacity = new_capacity;
        }
        __host__ __device__ void copy(const cpm::stack<ElemType> other, int copy_top_offset) {
            if (this->get_capacity() != other.get_capacity()) {
                Printer::stack_error("this capacity != other capacity when copy");
                return;
            }
            for (int i = 0; i < other.size - copy_top_offset; i++) {
                this->data[i] = other.data[i];
            }
        }
        __host__ __device__ void push_copy(ElemType value, int copy_top_offset = 0) {
#ifdef __CUDA_ARCH__
            if (!isFull()) {
                data[size].copy(value, copy_top_offset);
                size++;
            }
#else
            if (isFull()) {
                resize();
            }
            data[size].copy(value, copy_top_offset);
            size++;
#endif
        }

        __host__ __device__ void push(ElemType value) {
#ifdef __CUDA_ARCH__
            if (!isFull()) {
                data[size] = value;
                size += 1;
            }
            else {
                Printer::stack_error("stack is full, but tried to push");
            }
#else
            if (isFull()) {
                resize();
            }
            data[size] = value;
            size += 1;
#endif
        }

        __host__ __device__ ElemType pop() {
            if (size > 0) {
                return data[--size];
            }
        }

        __host__ __device__ ElemType* top_pointer() const {
            if (!isEmpty()) {
                return data + (size - 1);
            }
            Printer::stack_error("stack is empty, but tried to peek pointer");
            return nullptr;
        }
        __host__ __device__ ElemType top(int offset = 0) const {
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

            return data[size - 1 - offset];
        }

        __host__ __device__ ElemType* get_data() {
            return data;
        }

        __host__ __device__ size_t get_size() const {
            return size;
        }

        __host__ __device__ size_t get_capacity() const {
            return capacity;
        }

        __host__ __device__ void set_data(ElemType* new_data, int new_cap, int size = 0) {
            free(this->data);
            this->data = new_data;
            this->capacity = new_cap;
            this->size = size;
        }
        __host__ __device__ void set_size(int size) {
            if (size >= 0) {
                this->size = size;
            }
        }
    };
}