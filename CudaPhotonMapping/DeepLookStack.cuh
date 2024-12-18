#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "Stack.cuh"
#include "Printer.cuh"

namespace cpm
{
    template <typename T>
    class DeepLookStack : public cpm::stack<T>
    {
        int top = -1;

    public:
        __host__ __device__ DeepLookStack(size_t capacity) : cpm::stack<T>(capacity) {}
        __host__ __device__ void push(const T &elem)
        {
            cpm::stack<T>::push(elem);
            top++;
        }
        __host__ __device__ T peek(size_t offset = 0)
        {
            if (this->isEmpty())
            {
                Printer::deep_look_stack_error("stack is empty, but tried to peek");
            }
            if (top - offset < 0)
            {
                Printer::deep_look_stack_error("offset more then size, but tried to peek");
            }
            return this->data[top - offset];
        }
        __host__ __device__ void pop()
        {
            cpm::stack<T>::pop();
            top--;
        }
    };
}