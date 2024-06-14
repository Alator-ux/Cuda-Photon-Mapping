#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "Stack.cuh"
namespace cpm
{
    template <typename T>
    class DeepLookStack : public cpm::stack<T>
    {
        int top = -1;

    public:
        void push(const T &elem)
        {
            cpm::stack<T>::push(elem);
            top++;
        }
        T peek(size_t offset = 0)
        {
            if (this->isEmpty())
            {
                //printf("Error: Stack is empty\n");
                // throw std::exception("Stack is empty");
            }
            if (top - offset < 0)
            {
                //printf(top - offset + " < 0\n");
                // throw std::exception(top - offset + " < 0");
            }
            return this->data[top - offset];
        }
        void pop()
        {
            cpm::stack<T>::pop();
            top--;
        }
    };
}