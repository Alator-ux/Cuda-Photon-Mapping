#pragma once

#include <iostream>
#include <chrono>
#include <cuda_runtime.h>

class Timer {
    std::chrono::high_resolution_clock::time_point start_time_cpu;
    std::chrono::high_resolution_clock::time_point stop_time_cpu;

    cudaEvent_t cuda_start;
    cudaEvent_t cuda_stop;
public:
    Timer() : start_time_cpu(), stop_time_cpu(), cuda_start(nullptr), cuda_stop(nullptr) {
        // ������������� CUDA �������
        cudaEventCreate(&cuda_start);
        cudaEventCreate(&cuda_stop);
    }

    ~Timer() {
        // ����������� CUDA �������
        cudaEventDestroy(cuda_start);
        cudaEventDestroy(cuda_stop);
    }

    // ������ ��������� ��� CPU
    void startCPU() {
        start_time_cpu = std::chrono::high_resolution_clock::now();
    }

    // ����� ��������� ��� CPU
    void stopCPU() {
        stop_time_cpu = std::chrono::high_resolution_clock::now();
    }

    // ���������� ���������� ������� ��� CPU � �������������
    double elapsedCPU() const {
        std::chrono::duration<double, std::milli> elapsed = stop_time_cpu - start_time_cpu;
        return elapsed.count();
    }

    // ������ ��������� ��� CUDA
    void startCUDA() {
        cudaEventRecord(cuda_start, 0);
    }

    // ����� ��������� ��� CUDA
    void stopCUDA() {
        cudaEventRecord(cuda_stop, 0);
        cudaEventSynchronize(cuda_stop);
    }

    // ���������� ���������� ������� ��� CUDA � �������������
    float elapsedCUDA() const {
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, cuda_start, cuda_stop);
        return milliseconds;
    }

    // ����� ������� � �������
    void printCPU(const std::string& label = "CPU") const {
        std::cout << label << " Time: " << elapsedCPU() << " ms" << std::endl;
    }

    void printCUDA(const std::string& label = "CUDA") const {
        std::cout << label << " Time: " << elapsedCUDA() << " ms" << std::endl;
    }
};