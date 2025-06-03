//#pragma once
//#include <crt/host_defines.h>
//#include <cuda_runtime.h>
//#include "Defines.cuh"
//#include "Printer.cuh"
//#include "CudaCLinks.cuh"
//#include "GlobalParams.cuh"
//
//namespace cpm {
//    template<typename T>
//    class queue {
//    private:
//        T* data;
//        idxtype capacity;
//        idxtype head;
//        idxtype tail;
//        idxtype current_size;
//
//    public:
//        __device__ void init(T* data, idxtype capacity) {
//            this->data = data;
//            this->capacity = capacity;
//            this->current_size = 0;
//            this->head = 0;
//            this->tail = 0;
//        }
//
//        __host__ __device__ bool is_empty() const {
//            return current_size == 0;
//        }
//
//        __host__ __device__ bool is_full() const {
//            return current_size == capacity;
//        }
//
//        __host__ __device__ bool enqueue(const T& value) {
//            if (is_full()) {
//                return false;
//            }
//            oidxtype old_head_idx = atomicInc((idxtype*)&head, capacity);
//            data[old_head_idx] = value;
//            return true;
//        }
//
//        __host__ __device__ bool dequeue(T& out_value) {
//            if (is_empty()) {
//                return false;
//            }
//            atomicDec((idxtype*)&tail, );
//            out_value = data[head];
//            return true;
//        }
//
//        __host__ __device__ bool peek(T& out_value) const {
//            if (is_empty()) {
//                return false;
//            }
//            out_value = data[head];
//            return true;
//        }
//
//        __host__ __device__ idxtype size() const {
//            return current_size;
//        }
//
//        __host__ __device__ idxtype get_capacity() const {
//            return capacity;
//        }
//    };
//}