#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

namespace cpm {
	template<typename ElemType>
	class stack {
        ElemType* data;
        size_t size, capacity;

    public:
        __host__ __device__ stack(size_t capacity) {
            data = new ElemType[capacity];
            size = 0;
            this->capacity = capacity - 1;
        }

        __host__ __device__ ~stack() {
            delete[] data;
        }

        __host__ __device__ bool isEmpty() const {
            return size == 0;
        }

        __host__ __device__ bool isFull() const {
            return size >= capacity;
        }

        __host__ __device__ void push(ElemType value) {
            if (!isFull()) {
                data[size] = value;
                size += 1;
            }
        }

        __host__ __device__ void pop() {
            if (size > 0) {
                size -= 1;
            }
        }

        __host__ __device__ ElemType top() const {
            if (!isEmpty()) {
                return data[size - 1];
            }
            return ElemType();
        }
	};
}