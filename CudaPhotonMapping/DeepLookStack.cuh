//#pragma once
//#include "cuda_runtime.h"
//#include "device_launch_parameters.h"
//#include "Stack.cuh"
//#include "Printer.cuh"
//
//namespace cpm
//{
//    template <typename T>
//    class DeepLookStack : public cpm::array_stack<T>
//    {
//        int top = -1;
//
//    public:
//        __host__ __device__ DeepLookStack() : cpm::stack<T>() {}
//        __host__ __device__ DeepLookStack(size_t capacity) : cpm::stack<T>(capacity) {}
//        __host__ __device__ void push(const T &elem)
//        {
//            cpm::stack<T>::push(elem);
//            top++;
//        }
//        __host__ __device__ void push_copy(const T& elem)
//        { 
//            cpm::stack<T>::push_copy(elem);
//            top++;
//        }
//        __host__ __device__ T peek(int offset = 0)
//        {
//            if (this->isEmpty())
//            {
//                Printer::deep_look_stack_error("stack is empty, but tried to peek");
//                return T();
//            }
//            if (top - offset < 0)
//            {
//                Printer::deep_look_stack_error("offset more then size, but tried to peek");
//                return T();
//            }
//            return this->data[top - offset];
//        }
//        __host__ __device__ void pop()
//        {
//            cpm::stack<T>::pop();
//            top--;
//        }
//        __host__ __device__ void copy(const DeepLookStack<T> other) {
//            if (this->get_capacity() != other.get_capacity()) {
//                Printer::deep_look_stack_error("this capacity != other capacity when copy");
//                return;
//            }
//            for (int i = 0; i < other.top; i++) {
//                this->data[i] = other.data[i];
//            }
//        }
//
//        __host__ __device__ T* get_data() {
//            return cpm::stack<T>::get_data();
//        }
//
//    };
//}