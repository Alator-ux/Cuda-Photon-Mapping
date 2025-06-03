#pragma once
#include "cuda_runtime.h"
#include "Defines.cuh"
#include "CudaUtils.cuh"

struct PhotonMaxHeapItem {
	uint idx;
	float distance;

	__host__ __device__ PhotonMaxHeapItem(uint idx = 0, float dist = 1e30f) : idx(idx), distance(dist) {}

	__host__ __device__ bool operator<(const PhotonMaxHeapItem& other) {
		return distance < other.distance;
	}
};

extern PhotonMaxHeapItem* cpu_photon_heap_data;
extern __constant__ PhotonMaxHeapItem* gpu_photon_heap_data;

__host__ __device__ __forceinline__ PhotonMaxHeapItem* photon_heap_data() {
#ifdef __CUDA_ARCH__
    return gpu_photon_heap_data;
#else
    return cpu_photon_heap_data;
#endif
}

extern uint cpu_photon_heap_capacity;
extern __constant__ uint gpu_photon_heap_capacity;

__host__ __device__ __forceinline__ uint photon_heap_capacity() {
#ifdef __CUDA_ARCH__
    return gpu_photon_heap_capacity;
#else
    return cpu_photon_heap_capacity;
#endif
}

__host__ void set_photon_heap_parameters(PhotonMaxHeapItem* cpu_photon_heap_data_val, PhotonMaxHeapItem* gpu_photon_heap_data_val, uint photon_heap_capacity_val);

#define PHOTON_HEAP_OFFSET(array_index, array_capacity, idx) ((array_capacity) * (idx) + (array_index))

class PhotonMaxHeap {
    uint size;
    __host__ __device__ void sift_up(uint idx, PhotonMaxHeapItem* data, uint array_idx, uint array_cap) {
        uint idx_offset = PHOTON_HEAP_OFFSET(array_idx, array_cap, idx);
        while (idx > 0) {
            uint parent = (idx - 1) / 2;
            uint parent_offset = PHOTON_HEAP_OFFSET(array_idx, array_cap, parent);
            if (data[idx_offset].distance <= data[parent_offset].distance) {
                break;
            }
            cpm::swap(data[idx_offset], data[parent_offset]);
            idx = parent;
            idx_offset = parent_offset;
        }
    }

    __host__ __device__ void sift_down(uint idx, PhotonMaxHeapItem* data, uint array_idx, uint array_cap) {
        uint idx_offset = PHOTON_HEAP_OFFSET(array_idx, array_cap, idx);
        uint largest_offset = idx_offset;
        while (true) {
            uint largest = idx;
            uint left = 2 * idx + 1;
            uint left_offset = PHOTON_HEAP_OFFSET(array_idx, array_cap, left);
            uint right = 2 * idx + 2;
            uint right_offset = PHOTON_HEAP_OFFSET(array_idx, array_cap, right);

            if (left < size && data[left_offset].distance > data[largest_offset].distance) {
                largest = left;
                largest_offset = left_offset;
            }
            if (right < size && data[right_offset].distance > data[largest_offset].distance) {
                largest = right;
                largest_offset = right_offset;
            }

            if (largest == idx) {
                break;
            }
            cpm::swap(data[idx_offset], data[largest_offset]);
            idx = largest;
            idx_offset = largest_offset;
        }
    }
public:
    __host__ __device__ 
    PhotonMaxHeap() : size(0) {}

    __host__ __device__ void push(uint photon_idx, float distance, int num_to_find, uint array_idx, uint array_cap) {
        PhotonMaxHeapItem new_item(photon_idx, distance);

        PhotonMaxHeapItem* data = photon_heap_data();
        if (size < num_to_find) {
            uint offset = PHOTON_HEAP_OFFSET(array_idx, array_cap, size);
            data[offset] = new_item;
            size++;
            sift_up(size - 1, data, array_idx, array_cap);
        }
        else {
            uint offset = PHOTON_HEAP_OFFSET(array_idx, array_cap, 0);
            if (distance < data[offset].distance) {
                data[offset] = new_item;
                sift_down(0, data, array_idx, array_cap);
            }
        }
    }

    __host__ __device__ bool pop(PhotonMaxHeapItem& result, uint array_idx, uint array_cap) {
        if (size == 0) return false;

        PhotonMaxHeapItem* data = photon_heap_data();
        uint zero_offset = PHOTON_HEAP_OFFSET(array_idx, array_cap, 0);
        uint last_offset = PHOTON_HEAP_OFFSET(array_idx, array_cap, size - 1);
        result = data[zero_offset];
        data[zero_offset] = data[last_offset];
        size--;
        if (size > 0) {
            sift_down(0, data, array_idx, array_cap);
        }
        return true;
    }

    __host__ __device__ uint get_size() const { return size; }

    __host__ __device__ void clear() { size = 0; }
};