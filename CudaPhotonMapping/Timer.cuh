#pragma once

#include <iostream>
#include <chrono>
#include <cuda_runtime.h>

class Timer {
    std::chrono::high_resolution_clock::time_point start_time_cpu;
    std::chrono::high_resolution_clock::time_point stop_time_cpu;

    cudaEvent_t cuda_start;
    cudaEvent_t cuda_stop;

    float average_cuda_time;
    size_t cuda_count;
public:
    Timer() : start_time_cpu(), stop_time_cpu(), cuda_start(nullptr), cuda_stop(nullptr) {
        cudaError_t err;

        err = cudaEventCreate(&cuda_start);
        if (err != cudaSuccess) {
            std::cerr << "Failed to create start event: " << cudaGetErrorString(err) << std::endl;
            exit(-1);
        }
        err = cudaEventCreate(&cuda_stop);
        if (err != cudaSuccess) {
            std::cerr << "Failed to create start event: " << cudaGetErrorString(err) << std::endl;
            exit(-1);
        }

        resetAverageCUDA();
    }

    Timer(const Timer&) = delete;
    Timer& operator=(const Timer&) = delete;

    ~Timer() {
        cudaEventDestroy(cuda_start);
        cudaEventDestroy(cuda_stop);
    }

    // Начало измерения для CPU
    void startCPU() {
        start_time_cpu = std::chrono::high_resolution_clock::now();
    }

    // Конец измерения для CPU
    void stopCPU() {
        stop_time_cpu = std::chrono::high_resolution_clock::now();
    }

    // Вычисление прошедшего времени для CPU в миллисекундах
    double elapsedCPU() const {
        std::chrono::duration<double, std::milli> elapsed = stop_time_cpu - start_time_cpu;
        return elapsed.count();
    }

    // Начало измерения для CUDA
    void startCUDA() {
        cudaEventRecord(cuda_start, 0);
    }

    // Конец измерения для CUDA
    void stopCUDA() {
        cudaEventRecord(cuda_stop, 0);
        cudaEventSynchronize(cuda_stop);
        average_cuda_time += elapsedCUDA();
        cuda_count += 1;
    }

    void resetAverageCUDA() {
        cuda_count = 0;
        average_cuda_time = 0;
    }

    // Вычисление прошедшего времени для CUDA в миллисекундах
    float elapsedCUDA() const {
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, cuda_start, cuda_stop);
        return milliseconds;
    }

    // Вывод времени в консоль
    void printCPU(const std::string& label = "CPU") const {
        std::cout << label << " Time: " << elapsedCPU() << " ms" << std::endl;
    }

    void printCUDA(const std::string& label = "CUDA") const {
        std::cout << label << " Time: " << elapsedCUDA() << " ms" << std::endl;
    }

    void printAverageCUDA(const std::string& label = "CUDA Average") const {
        std::cout << label << " Time: " << average_cuda_time / cuda_count << " ms" << std::endl;
    }
};