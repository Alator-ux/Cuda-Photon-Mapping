#pragma once
#include "thrust/device_vector.h"
#include "thrust/host_vector.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
namespace cpm {

    template <typename ElemType, typename ComparatorType>
    class priority_queue {
    protected:
        ElemType* heapData;
        ComparatorType comparator;
    private:
        size_t _size, _capacity;
    

        __host__ __device__ void heapify_up(int index) {
            bool swap = true;
            while (index > 0 && swap) {
                int parentIndex = (index - 1) / 2;

                swap = comparator(&heapData[index], &heapData[parentIndex]);
                if (swap) {
                    thrust::swap(heapData[index], heapData[parentIndex]);
                    index = parentIndex;
                }
            }
        }

        __host__ __device__ void heapify_down(int index) {
            bool swap;
            bool heapify = true;
            while (heapify) {
                int leftChild = 2 * index + 1;
                int rightChild = 2 * index + 2;
                int largest = index;

                swap = comparator(&heapData[leftChild], &heapData[largest]);
                if (leftChild < _size && swap) {
                    largest = leftChild;
                }

                swap = comparator(&heapData[rightChild], &heapData[largest]);
                if (rightChild < _size && swap) {
                    largest = rightChild;
                }

                heapify = largest != index;
                if (heapify) {
                    thrust::swap(heapData[index], heapData[largest]);
                    index = largest;
                }
            }
        }

    public:
        __host__ __device__ priority_queue() : comparator(){}
        __host__ __device__ priority_queue(size_t capacity) : comparator() {
            heapData = new ElemType[capacity];
            _size = 0;
            _capacity = capacity;
        }
        __host__ __device__ priority_queue(ElemType* data, size_t capacity) : comparator() {
            heapData = data;
            _size = 0;
            _capacity = capacity;
        }


        __host__ __device__ void push(const ElemType& value) {
            heapData[_size] = value;
            _size++;
            heapify_up(_size - 1);
        }

        __host__ __device__ void pop() {
            if (_size == 1) {
                _size = 0;
                return;
            }
            if (!empty()) {
                thrust::swap(heapData[0], heapData[_size - 1]);
                _size--;
                heapify_down(0);
            }
        }

        __host__ __device__ ElemType top() const {
            if (!empty()) {
                return heapData[0];
            }
        }
        
        __host__ __device__ bool empty() const {
            return _size == 0;
        }

        __host__ __device__ size_t size() const {
            return _size;
        }
        __device__ void free_device() {
            delete[] heapData;
        }
    };
}