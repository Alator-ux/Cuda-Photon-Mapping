#include "PhotonMaxHeap.cuh"

PhotonMaxHeapItem* cpu_photon_heap_data;
__constant__ PhotonMaxHeapItem* gpu_photon_heap_data;

uint cpu_photon_heap_capacity;
__constant__ uint gpu_photon_heap_capacity;

__host__ void set_photon_heap_parameters(PhotonMaxHeapItem* cpu_photon_heap_data_val, PhotonMaxHeapItem* gpu_photon_heap_data_val, uint photon_heap_capacity_val) {
    cpu_photon_heap_data = cpu_photon_heap_data_val;
    cpu_photon_heap_capacity = photon_heap_capacity_val;

    cudaMemcpyToSymbol(gpu_photon_heap_data, &gpu_photon_heap_data_val, sizeof(PhotonMaxHeapItem*));
    checkCudaErrors(cudaGetLastError());
    cudaMemcpyToSymbol(gpu_photon_heap_capacity, &photon_heap_capacity_val, sizeof(uint));
    checkCudaErrors(cudaGetLastError());
}