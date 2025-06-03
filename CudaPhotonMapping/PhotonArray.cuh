#pragma once
#include <crt/host_defines.h>
#include <cuda_runtime.h>
#include "Printer.cuh"
#include "Defines.cuh"
#include "Photon.cuh"


namespace cpm {
	class PhotonArray {
		Photon* data;
		volatile idxtype size;
		idxtype capacity;


	public:
		__host__ __device__
			PhotonArray(Photon* data, idxtype size, idxtype capacity) : data(data), size(size), capacity(capacity) {}

		__host__ __device__
			PhotonArray() : data(nullptr), size(0), capacity(0) {}

		__host__ __device__
			void init() {
			data = nullptr;
			size = 0;
			capacity = 0;
		}

		__host__ __device__
		idxtype add(Photon value) {
#ifdef __CUDA_ARCH__
			idxtype idx = atomicAdd((idxtype*)&size, 1);
#else
			idxtype idx = size;
#endif
			if (idx + 1 > capacity) {
				return IDXTYPE_NONE_VALUE;
			}
			data[idx] = value;
			return idx;
		}

		__host__ __device__ bool is_not_full() {
			return size < capacity;
		}

		__host__ __device__
		Photon& operator[](idxtype ind) {
			if (ind >= size) {
				Printer::index_out_of_bound_error("class List");
			}
			
			return ((Photon*)data)[ind];
		}

		__host__ __device__
		idxtype get_size() { return size; }

		__host__ __device__
		idxtype get_capacity() { return capacity; }

		__host__ __device__
		Photon* get_data() { return (Photon*)data; }
	};
}