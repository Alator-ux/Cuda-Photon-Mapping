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

        __host__ __device__ size_t getSize() const {
            return size;
        }

        __host__ __device__ size_t getCapacity() const {
            return capacity;
        }

        __host__ __device__ bool isEmpty() const {
            return size == 0;
        }

        __host__ __device__ bool isFull() const {
            return size > capacity;
        }

        __host__ __device__ void push(ElemType value) {
#ifdef __CUDA_ARCH__
            if (!isFull()) {
                data[size] = value;
                size += 1;
            }
#else
            if (isFull()) {
                size_t new_capacity = capacity == 0 ? 1 : capacity * 2;
                ElemType* new_data = (ElemType*)malloc(new_capacity * sizeof(ElemType));

                for (size_t i = 0; i < capacity; i++) {
                    new_data[i] = data[i];
                }

                free((void*)data);
                data = new_data;
                capacity = new_capacity;
            }
            data[size] = value;
            size += 1;
#endif
        }

        __host__ __device__ void pop() {
            if (size > 0) {
                size -= 1;
            }
        }

        __host__ __device__ ElemType* top_pointer() const {
            if (!isEmpty()) {
                return data + (size - 1);
            }
            return nullptr;
        }

        __host__ __device__ ElemType top() const {
            if (!isEmpty()) {
                return data[size - 1];
            }
            return ElemType();
        }
	};
}