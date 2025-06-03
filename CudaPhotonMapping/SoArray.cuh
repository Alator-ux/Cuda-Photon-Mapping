#pragma once
#include <crt/host_defines.h>
#include <cuda_runtime.h>
#include "Defines.cuh"
#include "Printer.cuh"
#include "CudaCLinks.cuh"
#include "GlobalParams.cuh"

namespace cpm {
	template <typename T, typename TContent>
	class SoArray {
		T data;
		volatile idxtype size;
		idxtype capacity;

		SoArray<T, TContent>* device_pointer = nullptr;

	public:
		__host__ __device__
		SoArray(T data, idxtype size, idxtype capacity) : data(data), size(size), capacity(capacity) {}

		__host__ __device__
		SoArray() : data(T()), size(0), capacity(0) {}

		__host__ __device__
		idxtype add(const TContent& content) {
			idxtype idx = atomicAdd((idxtype*)&size, 1);
			if (idx > capacity) {
				printf("Error, out of capacity in cpm::SoArray");
			}
			data.add(idx, content);
			return idx;
		}

		/*__host__
		idxtype add_from_host_to_device(T value) {
			cudaMemcpy(data + size, &value, sizeof(T), cudaMemcpyHostToDevice);
			size += 1;
			cudaMemcpy(device_pointer, this, sizeof(Array<T>), cudaMemcpyHostToDevice);
			return size - 1;
		}*/

		__host__
		void initialize_on_device(idxtype capacity) {
			this->data.free_fields();
			cudaFree(this->device_pointer);


			this->capacity = capacity;
			this->size = 0;

			this->data.initialize_on_device(capacity);

			cudaMalloc(&device_pointer, sizeof(SoArray<T, TContent>));
			cudaMemcpy(device_pointer, this, sizeof(SoArray<T, TContent>), cudaMemcpyHostToDevice);
		}

		__host__
		idxtype get_size_from_device_to_host() {
			SoArray<T, TContent>* host_array = (SoArray<T, TContent>*)malloc(sizeof(SoArray<T, TContent>));
			cudaMemcpy(host_array, device_pointer, sizeof(SoArray<T, TContent>), cudaMemcpyDeviceToHost);

			this->size = host_array->size;

			free(host_array);

			return this->size;
		}

		__host__
		void free_from_device() {
			SoArray<T, TContent> this_pointer;
			cudaMemcpy(&this_pointer, device_pointer, sizeof(SoArray<T, TContent>), cudaMemcpyDeviceToHost);
			this_pointer.data.free_fields();
			cudaFree(device_pointer);
		}

		__host__ __device__
		idxtype get_size() { return size; }

		__host__ __device__
		void set_size(idxtype new_size) { size = new_size; }

		__host__ __device__
		idxtype get_capacity() { return capacity; }

		__host__ __device__
		T get_data() { return data; }

		__host__ __device__
		SoArray<T, TContent>* get_device_pointer() { return device_pointer; }

	};
}