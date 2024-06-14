#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "Stack.cuh"
#include "thrust/swap.h"
namespace cpm
{
    template <typename T, typename Comparator>
    __host__ __device__ int partition(T* arr, int low, int high, Comparator comp) {
        T pivot = arr[high];
        int i = low - 1;

        for (int j = low; j <= high - 1; j++) {
            if (comp(arr[j], pivot)) {
                i++;
                thrust::swap(arr[i], arr[j]);
            }
        }
        thrust::swap(arr[i + 1], arr[high]);
        return (i + 1);
    }

    template <typename T, typename Comparator>
    __host__ __device__ void quick_sort(T* arr, int low, int high, Comparator comp) {
        high -= 1;
        // Создаем стек для хранения начальных и конечных индексов подмассивов
        auto stack = cpm::stack<int>(high - low + 1);
        stack.push(low);
        stack.push(high);

        while (!stack.isEmpty()) {
            high = stack.top();
            stack.pop();
            low = stack.top();
            stack.pop();
            int p = partition<T, Comparator>(arr, low, high, comp);

            if (p - 1 > low) {
                stack.push(low);
                stack.push(p - 1);
            }

            if (p + 1 < high) {
                stack.push(p + 1);
                stack.push(high);
            }
        }
    }
}