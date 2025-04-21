#include "DynamicArrayTest.cuh"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "List.cuh"
#include "CudaGridSynchronizer.cuh"
#include "Defines.cuh"
#include "Timer.cuh"
#include <string>
#include <format>
#include <iostream>
#include <sstream>
#include "Initializer.cuh"
#include "PrefixScanSum.cuh"
#include <cooperative_groups.h>

using namespace cpm;

__global__ void list_init_kernel(List<int>* list) {
    list->init();
}
__device__ volatile uint waiting_blocks = 0;
__device__ volatile uint test_sum = 0;

//__device__ void proccess_if_waiting(List<int>* list) {
//    if (threadIdx.x == 0) {
//        atomicInc((unsigned int*)&waiting_blocks, gridDim.x - 1);
//    }
//    __syncthreads();
//    while (waiting_blocks != 0) {
//        list->process_status_code();
//    }
//}

__global__ void list_add_kernel(List<int>* list, uint test_times) {
    namespace cg = cooperative_groups;
    auto grid = cg::this_grid();

    for (uint i = 0; i < test_times; i++) {
        list->add(threadIdx.x, grid, threadIdx.x == 0 && blockIdx.x == 0);
        list->add(threadIdx.x, grid, threadIdx.x % 2 == 0);
        list->add(threadIdx.x, grid, blockIdx.x == 0);
        list->add(threadIdx.x < 5 ? 0 : 1, grid, true);
    }
}

/*
* Inserted 34305 elements to list
   Time: 90.5851 ms
*/
__global__ void list_insert_kernel(List<int>* list, uint test_times) {
    namespace cg = cooperative_groups;
    auto grid = cg::this_grid();

    list->add(-1, grid, threadIdx.x == 0 && blockIdx.x == 0);
    for (uint i = 0; i < test_times; i++) {
        list->insert(0, 1, grid, threadIdx.x == 0 && blockIdx.x == 0);
        list->insert(0, 441, grid, threadIdx.x % 2 == 0);
        list->insert(0, 888, grid, blockIdx.x == 0);
        //list->insert(0, 441, grid);
    }
}

__global__ void list_print_kernel(List<int>* list, uint expected_size, bool print_only_size) {
    auto local_list = *list;
    if (!print_only_size) {
        printf("\nPrinting list\n");
        for (uint i = 0; i < local_list.get_size(); i++) {
            printf("%i ", local_list[i]);
        }
    }
    printf("\nTotal size: %u, expected: %u\n", local_list.get_size(), expected_size);
}

__global__ void list_equality_check_kernel(List<int>* flist, List<int>* slist) {
    auto flocal_list = *flist;
    auto slocal_list = *slist;
    if (flocal_list.get_size() != slocal_list.get_size()) {
        printf("Differrent sizes\n");
    }
    for (size_t i = 0; i < flocal_list.get_size(); i++) {
        if (flocal_list[i] != slocal_list[i]) {
            printf("Differrent values at %llu. Excpected %llu, but found %llu\n", i, flocal_list[i], slocal_list[i]);
        }
    }
}

__device__ PrescanHelperStruct<int> pshs;
__device__ int global_sum_offset;

__global__ void pss_kernel(int* arr, int length) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        pshs = get_prescan_helper_struct(arr, length, (int*)&global_sum_offset);
    }
    CudaGridSynchronizer::synchronize_grid();
    auto local_pshs = pshs;
    prescan(local_pshs, length, &global_sum_offset);
}

__global__ void pss_check_kernel(int* arr, int length) {    
    free((void*)pshs.separated_sums_arr);
    free((void*)pshs.united_sums_arr);
    size_t sum = 0;
    for (idxtype i = 0; i < length; i++) {
        if (sum != pshs.out_arr[i]) {
            printf("error in %llu / %i; ", sum, pshs.out_arr[i]);
        }
        sum += arr[i];
    }
    
}

void PrefixScanSumTest() {
    Timer timer;
    std::stringstream ss;

    Printer::kernel_properties(pss_kernel); // was regs 72, now 62

    //const int length = 1024+32+3;
    const int length = 1000000;
    int* arr_cpu = (int*)malloc(sizeof(int) * length);
    for (int i = 0; i < length; i++) {
        arr_cpu[i] = i % 7 ? (i % 2 + 1) % 2 : i % 2;
    }

    int* arr_gpu;
    cudaMalloc((void**)&arr_gpu, sizeof(int) * length);
    cudaMemcpy(arr_gpu, arr_cpu, sizeof(int) * length, cudaMemcpyHostToDevice);

    int threads = PRESCAN_THREADS;
    int blocks = PRESCAN_BLOCKS;
    int shared_mem = threads * 2 * sizeof(int);  // 512 * 2 * 4 = ~4 kb
    timer.startCUDA();
    pss_kernel << <blocks, threads,  shared_mem>>> (arr_gpu, length);
    timer.stopCUDA();  // was 0.42, now 0.33 (blocks cause of regs)
    checkCudaErrors(cudaGetLastError());
    ss << "\nPrefix scan alg processed " << length << " elements\n";
    timer.printCUDA(ss.str());

    pss_check_kernel << <1, 1 >> > (arr_gpu, length);
    CudaSynchronizer::synchronize_with_instance();
    checkCudaErrors(cudaGetLastError());
}

struct SuperData {
    int values[4];
    /* regs
    * 32 - 16
    * 16 - 25
    * 4  - 14
    */
};
__device__ SuperData sd;
__global__ void superkernel(int* sum) {
    SuperData local_sd = sd;
    size_t sz = 100000000;
    auto a = new size_t[sz];
    auto b = new uint[sz];
    /*for (int i = 0; i < 3; i++) {
        *sum += local_sd.values[i];
    }*/
    for (int i = 0; i < sz;i++) {
        printf("a + b = %i", a[i] + b[i]);
    }
}

void DynamicArrayTest() {
    size_t newHeapSize = 128 * 1024 * 1024; // 128 MB
    cudaDeviceSetLimit(cudaLimitMallocHeapSize, newHeapSize);

    PrefixScanSumTest();

    Printer::kernel_properties(list_insert_kernel);
    //Printer::kernel_properties(list_add_kernel);
    //Printer::kernel_properties(superkernel);
    //superkernel << <1, 16 >>> (nullptr);

    Timer timer;
    std::stringstream ss;
    void* args[2];

    List<int>* list, *other_list;
    cudaMalloc(&list, cpm::List<int>::gpu_sizeof());
    cudaMalloc(&other_list, cpm::List<int>::gpu_sizeof());
    list_init_kernel << <1, 1 >> > (list);
    CudaSynchronizer::synchronize_with_instance();
    checkCudaErrors(cudaGetLastError());


    uint threads = 512;
    uint blocks = 44;
    initialize_device_params(threads, blocks);
    uint expected_list_size = 1 +                    // first threads only
                              threads * blocks / 2 + // even threads only
                              threads +              // first block only
                              threads * blocks;      // all threads
    uint test_times = 3430500 / expected_list_size;
    expected_list_size *= test_times;

    //timer.startCUDA();
    ////list_add_kernel << <blocks, threads >> > (list, test_times);
    //args[0] = &list;
    //args[1] = &test_times;
    //checkCudaErrors(cudaLaunchCooperativeKernel((void*)list_add_kernel, blocks, threads, args, 0, nullptr));
    //timer.stopCUDA();
    //checkCudaErrors(cudaGetLastError());
    //ss.clear();
    //ss << "\nAdded " << expected_list_size << " elements to list \n";
    //timer.printCUDA(ss.str());

    //list_print_kernel << <1, 1 >> > (list, expected_list_size, true);
    //CudaSynchronizer::synchronize_with_instance();
    //checkCudaErrors(cudaGetLastError());

    test_times = 1;
    expected_list_size = 1 +                    // first threads only
                         threads * blocks / 2 + // even threads only
                         threads +              // first block only
                         threads * blocks;      // all threads;
    expected_list_size *= test_times;

    timer.startCUDA();
    args[0] = &list;
    args[1] = &test_times;
    checkCudaErrors(cudaLaunchCooperativeKernel((void*)list_insert_kernel, blocks, threads, args, 0, nullptr));
    timer.stopCUDA();
    checkCudaErrors(cudaGetLastError());
    ss.clear();
    ss << "\nInserted " << expected_list_size << " elements to list \n";
    timer.printCUDA(ss.str());

    list_print_kernel << <1, 1 >> > (list, expected_list_size, false);
    CudaSynchronizer::synchronize_with_instance();
    checkCudaErrors(cudaGetLastError());
}