#pragma once
#include <crt/host_defines.h>
#include <cuda_runtime.h>
#include "Defines.cuh"
#include "Printer.cuh"
#include "CudaCLinks.cuh"
#include "GlobalParams.cuh"

namespace cpm {
	template <typename T>
	class Array {
		T* data;
		volatile idxtype size;
		idxtype capacity;

		Array<T>* device_pointer = nullptr;

	public:
		__host__ __device__
		Array(T* data, idxtype size, idxtype capacity) : data(data), size(size), capacity(capacity) {}

		__host__ __device__
		Array() : data(nullptr), size(0), capacity(0) {}

		__host__ __device__
		idxtype add(T value) {
#ifdef __CUDA_ARCH__
			
			idxtype	idx = atomicAdd((idxtype*)&size, 1);

			if (size > capacity) {
				printf("Error, out of capacity in cpm::Array");
			}

			data[idx] = value;

			return idx;
#else
			idxtype idx = atomicAdd((idxtype*)&size, 1);
			if (size > capacity) {
				printf("Error, out of capacity in cpm::Array");
			}
			data[idx] = value;
			return idx;
#endif
		}

		__host__
		idxtype add_from_host_to_device(T value) {
			cudaMemcpy(data + size, &value, sizeof(T), cudaMemcpyHostToDevice);
			size += 1;
			cudaMemcpy(device_pointer, this, sizeof(Array<T>), cudaMemcpyHostToDevice);
			return size - 1;
		}

		__host__
		void initialize_on_device(idxtype capacity) {
			cudaFree(this->data);
			cudaFree(this->device_pointer);

			this->capacity = capacity;
			this->size = 0;

			cudaMalloc(&this->data, sizeof(T) * capacity);

			cudaMalloc(&device_pointer, sizeof(Array<T>));
			cudaMemcpy(device_pointer, this, sizeof(Array<T>), cudaMemcpyHostToDevice);
		}

		__host__
		idxtype get_size_from_device_to_host() {
			Array<T>* host_array = (Array<T>*)malloc(sizeof(Array<T>));
			cudaMemcpy(host_array, device_pointer, sizeof(Array<T>), cudaMemcpyDeviceToHost);

			this->size = host_array->size;

			free(host_array);

			return this->size;
		}

		template<bool on_device = true>
		__host__
		void fill(T (*f)(idxtype)) {
			T* host_data = new T[capacity];
			for (idxtype i = 0; i < capacity; i++) {
				host_data[i] = f(i);
			}
			if (on_device) {
				cudaMemcpy(data, host_data, sizeof(T) * capacity, cudaMemcpyHostToDevice);
				delete[] host_data;
			}
			else {
				delete[] data;
				data = host_data;
			}
		}

		__host__
		void free_device_from_host() {
			cudaFree(data);
			cudaFree(device_pointer);
			checkCudaErrors(cudaDeviceSynchronize());
		}

		__host__ __device__
		idxtype get_size() { return size; }

		__host__ __device__
		idxtype get_capacity() { return capacity; }

		__host__ __device__
		T* get_data() { return (T*)data; }

		__host__ __device__
		Array<T>* get_device_pointer() { return device_pointer; }

	};
}