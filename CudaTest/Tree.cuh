#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "PriorityQueue.cuh"
namespace cpm {
	template<typename ElemType>
    class Tree {
        ElemType** heap;
        size_t size, capacity, depth;
        __host__ __device__ int add(int child_ind, ElemType* child) {
            if (child_ind < 0 || child_ind> capacity) {
                return -1;
            }
            heap[child_ind] = child;
            size += 1;
            return child_ind;
        }
    public:
        __host__ __device__ Tree(ElemType** heap, size_t size, size_t capacity) {
            this->heap = heap;
            this->size = 0;
            this->capacity = capacity - 1;
            depth = hceil(log2f(capacity + 1)) - 1;
        }
        __host__ __device__ Tree(size_t capacity) {
            this->heap = (ElemType**)malloc(capacity * sizeof(ElemType*));
            this->size = 0;
            this->capacity = capacity - 1;
            depth = hceil(log2f(capacity + 1)) - 1;
        }
        __host__ __device__ int add_left(int parent, ElemType* child) {
            int left = parent * 2 + 1;
            return add(left, child);
        }
        __host__ __device__ int add_right(int parent, ElemType* child) {
            int right = parent * 2 + 2;
            return add(right, child);
        }
        __host__ __device__ ElemType* get_left(int parent) {
            return parent * 2 + 1;
        }
        __host__ __device__ ElemType* get_right(int parent) {
            return parent * 2 + 2;
        }
        /*__host__ __device__ int* get_traversal_order() {
            int double_depth = 2 * depth;
            cpm::priority_queue<int, >
        }*/
    };
}