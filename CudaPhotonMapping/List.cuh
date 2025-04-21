#pragma once
#include <crt/host_defines.h>
#include <cuda_runtime.h>
#include "Defines.cuh"
#include "Printer.cuh"
#include "CudaCLinks.cuh"
#include "GlobalParams.cuh"
#include <cooperative_groups.h>

// God forgive me for list structer on gpu
namespace cpm {

	// List aka Dynamic Array
	template <typename T, bool use_cooperative_groups = true>
	class List {
		static constexpr int status_code_none		= 0;
		static constexpr int status_code_add		= 1;
		static constexpr int status_code_resize		= 2;
		static constexpr int status_code_shift_data = 3;
		static constexpr int status_code_insert     = 4;

		volatile T* data;
		volatile idxtype size;
		volatile idxtype capacity;

		volatile int status_code = 0;
		volatile int status_code_after_operation = 0;
		idxtype shift_index = IDXTYPE_NONE_VALUE;

#ifdef __CUDA_ARCH__
		/* Only for device code */
		volatile T* new_data;
		volatile T* insert_arr;
		volatile int resize_flag = 1;
		volatile uint threads_to_wait = 0;
		volatile uint waiting_operation_threads = 0;
#endif

		template <typename CopyDataType>
		__device__ __forceinline__
		void copy_data(CopyDataType* source, CopyDataType* dest, idxtype size) {
			for (idxtype i = threadIdx.x + blockIdx.x * blockDim.x; i < size; i += gridDim.x * blockDim.x) {
				dest[i] = source[i];
			}
		}

		__host__ __device__
		void resize(LIST_CUDA_INNER_PARAMS_DEF) {
#ifdef __CUDA_ARCH__
			idxtype new_capacity;
			if (threadIdx.x == 0 && blockIdx.x == 0) {
				//printf("resize. old cap = %llu.\n", capacity);
				new_capacity = capacity == 0 ? 2 : capacity * 2;
				if (new_capacity < size) {
					new_capacity = size;
				}
				free((T*)insert_arr);
				new_data = new T[new_capacity];
				if (new_data == NULL) {
					printf("Error, not enough memory for list data allocation");
				}
				__threadfence();
				atomicExch((int*)&resize_flag, 0);
			}
			
			
			grid.sync();

			/*for (idxtype i = threadIdx.x + blockIdx.x * blockDim.x; i < capacity; i += gridDim.x * blockDim.x) {
				new_data[i] = data[i];
			}*/
			copy_data<T>((T*)data, (T*)new_data, capacity);
			
			grid.sync();
			

			//CudaGridSynchronizer::synchronize_grid();

			if (threadIdx.x == 0 && blockIdx.x == 0) {
				free((T*)data);
				data = (T*)new_data;
				capacity = new_capacity;

				new_data = new T[new_capacity];
				if (new_data == NULL) {
					printf("Error, not enough memory for list data allocation");
				}
				insert_arr = new_data;

				atomicExch((int*)&resize_flag, 1);
				atomicExch((int*)&status_code, status_code_after_operation);
			}
			

			grid.sync();
			//CudaGridSynchronizer::synchronize_grid();
#else
			capacity = capacity == 0 ? 2 : capacity * 2;
			T* new_data = new T[capacity];

			for (idxtype i = 0; i < size; i++) {
				new_data[i] = data[i];
			}

			free(data);
			data = new_data;
#endif 
		}
		
		__host__ __device__
		void shift_data_v1_1(idxtype local_shift_index, idxtype old_size, T* local_data, T* local_insert_arr) {
#ifdef __CUDA_ARCH__			

			for (idxtype i = threadIdx.x + blockIdx.x * blockDim.x; i < local_shift_index; i += gridDim.x * blockDim.x) {
				local_insert_arr[i] = local_data[i];
			}
			
			if (threadIdx.x == 0 && blockIdx.x == 0) {
				if (old_size > size) {
					printf("old_size = %u ", old_size);
				}
			}
			for (idxtype i = local_shift_index + threadIdx.x + blockIdx.x * blockDim.x; i < old_size - 1; i += gridDim.x * blockDim.x) {
				local_insert_arr[i + 1] = local_data[i];
			}


			//idxtype current_index = threadIdx.x + blockIdx.x * blockDim.x;
			//idxtype step = gridDim.x * blockDim.x;
			//idxtype shift_from = local_shift_index; // copy to local memory
			//idxtype last_index = size - 1;
			//if (current_index <= last_index) {
			//	current_index = last_index - current_index;
			//	while (current_index >= shift_from) {
			//		local_insert_arr[current_index + 1] = local_data[current_index];
			//		if (current_index < step) { // current_index - step < 0
			//			break;
			//		}
			//		current_index -= step;
			//	}
			//}
#endif
		}


	public:
		__host__ __device__
		List(T* data, idxtype size, idxtype capacity) : data(data), size(size), capacity(capacity) {}

		__host__ __device__
		List() : data(nullptr), size(0), capacity(0), shift_index(IDXTYPE_NONE_VALUE) {}

		__host__ __device__
		void init() {
			data = nullptr;
			size = 0;
			capacity = 0;
			shift_index = IDXTYPE_NONE_VALUE;
#ifdef __CUDA_ARCH__
			waiting_operation_threads = 0;
#endif
		}

		__host__ __device__
		idxtype add(T value, LIST_CUDA_OUTER_PARAMS_DEF) {
#ifdef __CUDA_ARCH__
			int idx;
			if (should_execute) {
				idx = atomicAdd((idxtype*)&size, 1);
			}

			grid.sync();
			//CudaGridSynchronizer::synchronize_grid();

			if (size > capacity) {
				resize(grid);
			}

			if (should_execute) {
				data[idx] = value;
			}

			return should_execute ? idx : IDXTYPE_NONE_VALUE;
#else
			idxtype idx = atomicAdd((idxtype*)&size, 1);
			if (size > capacity) {
				resize(grid);
			}
			data[idx] = value;
			return idx;
#endif
		}

		__host__ __device__
		idxtype insert(idxtype ind, T value, LIST_CUDA_OUTER_PARAMS_DEF) {
#ifdef __CUDA_ARCH__
			idxtype old_size = size;

			grid.sync();

			if (should_execute) {
				atomicAdd((uint*)&waiting_operation_threads, 1);
				atomicAdd((idxtype*)&size, 1);
			}

			//CudaGridSynchronizer::synchronize_grid();
			grid.sync();

			if (size > capacity) {
				resize(grid);
			}

			idxtype local_shift_index = IDXTYPE_NONE_VALUE;

			bool inserting = false;
			while (waiting_operation_threads > 0) {
				if (should_execute) {
					idxtype old_shift_index = atomicCAS((idxtype*)&shift_index, IDXTYPE_NONE_VALUE, ind);
					if (old_shift_index == IDXTYPE_NONE_VALUE) {
					    atomicSub((uint*)&waiting_operation_threads, 1);
						inserting = true;
						local_shift_index = ind;
						//printf("%u ", ind);
					}
					else if (old_shift_index <= ind) {
						ind += 1;
						local_shift_index = old_shift_index;
					}
				}
				grid.sync();
				if (!should_execute) {
					local_shift_index = shift_index;
				}
				old_size += 1;
				__threadfence();
				shift_data_v1_1(local_shift_index, old_size, (T*)data, (T*)insert_arr);


				if (should_execute && inserting) {
					insert_arr[ind] = value;
					atomicExch((idxtype*)&shift_index, IDXTYPE_NONE_VALUE);
					should_execute = false;

					T* temp_pointer = (T*)data;
					data = insert_arr;
					insert_arr = temp_pointer;
				}

				grid.sync();
			}

			return inserting ? ind : IDXTYPE_NONE_VALUE;
#endif
		}

		

		__host__ __device__
		void set_at(T value, idxtype ind) {
			if (ind >= size) {
				Printer::index_out_of_bound_error("class List");
				return T();
			}
			data[ind] = value;
		}

		//__host__ __device__
		//bool process_status_code() {
		//	bool waiting = false;
		//	if (status_code == status_code_resize) {
		//		resize();
		//		waiting = true;
		//	}
		//	else if (status_code == status_code_shift_data) {
		//		//shift_data();
		//		waiting = true;
		//	}
		//	else if (status_code == status_code_add) {
		//		waiting = true;
		//	}
		//	return waiting;
		//}

		__host__ __device__
		T& operator[](idxtype ind) {
			if (ind >= size) {
				Printer::index_out_of_bound_error("class List");
				//return T();
			}
			/*while (status_code != status_code_none) {
				process_status_code();
			}*/
			return ((T*)data)[ind];
		}

		__host__ __device__
		idxtype get_size() { return size; }

		__host__ __device__
		idxtype get_capacity() { return capacity; }

		__host__
		static uint gpu_sizeof() {
			return sizeof(T*) + sizeof(idxtype) * 3 + sizeof(int) * 2 + // common part
				sizeof(T*) + sizeof(idxtype*) + sizeof(int) + sizeof(uint) * 2; // gpu part
		}
	};
}