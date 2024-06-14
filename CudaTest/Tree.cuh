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
            depth = ceilf(log2f(capacity + 1)) - 1;
        }
        __host__ __device__ Tree(size_t capacity) {
            this->heap = (ElemType**)malloc(capacity * sizeof(ElemType*));
            this->size = 0;
            this->capacity = capacity - 1;
            depth = ceilf(log2f(capacity + 1)) - 1;
        }
        __host__ __device__ ~Tree() {
            // TODO
        }
       /* template<typename Comparator>
        __host__ __device__ void fill_balanced()*/
        __host__ __device__ int set_root(ElemType* value) {
            return add(0, value);
        }
        __host__ __device__ int set_at(int index, ElemType* value) {
            return add(index, value);
        }
        __host__ __device__ int add_left(int parent, ElemType* child) {
            int left = parent * 2 + 1;
            return add(left, child);
        }
        __host__ __device__ int add_right(int parent, ElemType* child) {
            int right = parent * 2 + 2;
            return add(right, child);
        }
        __host__ __device__ bool has_root() {
            return heap[0] != nullptr;
        }
        __host__ __device__ bool has_left(int parent) {
            int left = parent * 2 + 1;
            if (left > capacity) {
                return false;
            }
            return heap[left] != nullptr;
        }
        __host__ __device__ bool has_right(int parent) {
            int right = parent * 2 + 2;
            if (right > capacity) {
                return false;
            }
            return heap[right] != nullptr;
        }
        __host__ __device__ bool is_leaf(int index) {
            return !(has_left(index) || has_right(index));
        }
        __host__ __device__ bool is_siblings(int index, int other) {
            if (index % 2 == 1) {
                return other - index == 1;
            }
            return index - other == 1;
        }
        __host__ __device__ ElemType* get_root() {
            if (has_root()) {
                return heap[0];
            }
            return nullptr;
        }
        __host__ __device__ ElemType* get_left(int parent) {
            if (has_left(parent)) {
                return heap[parent * 2 + 1];
            }
            return nullptr;
        }
        __host__ __device__ ElemType* get_right(int parent) {
            if (has_right(parent)) {
                return heap[parent * 2 + 2];
            }
            return nullptr;
        }
        __host__ __device__ int get_left_ind(int parent) {
            int left = parent * 2 + 1;
            if (left > capacity) {
                left = -1;
            }
            return left;
        }
        __host__ __device__ int get_right_ind(int parent) {
            int right = parent * 2 + 2;
            if (right > capacity) {
                right = -1;
            }
            return right;
        }
        __host__ __device__ int get_capacity() {
            return capacity + 1;
        }
        __host__ __device__ int get_depth() {
            return depth;
        }
        /*__host__ __device__ int* get_traversal_order() {
            int double_depth = 2 * depth;
            cpm::priority_queue<int, >
        }*/
    };
}