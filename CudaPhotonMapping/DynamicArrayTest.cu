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
#include "CGPrefixScanSum.cuh"

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
__global__ 
#ifdef __CUDA_ARCH__
__launch_bounds__(PRESCAN_THREADS, 2)
#endif
void list_insert_kernel(List<int>* list, uint test_times) {
    namespace cg = cooperative_groups;
    auto grid = cg::this_grid();

    list->add(-1, grid, threadIdx.x == 0 && blockIdx.x == 0);
    for (uint i = 0; i < test_times; i++) {

        list->insert(0, 1, grid, threadIdx.x == 0 && blockIdx.x == 0);
        list->insert(0, 4, grid, threadIdx.x % 2 == 0);
        list->insert(0, 8, grid, blockIdx.x == 0);
        list->insert(0, 7, grid);
        list->insert(0, 7, grid);
        list->insert(0, 7, grid);
    }
}

__global__ void list_print_kernel(List<int>* list, uint expected_size, bool print_only_size) {
    auto local_list = list->get_data();
    auto sz = list->get_size();
    if (!print_only_size) {
        /*printf("\nPrinting list\n");
        for (uint i = 0; i < sz; i++) {
            printf("%i ", local_list[i]);
        }*/
        printf("\nPrinting list\n");
        uint to = 0;
        uint from;
        to = PRESCAN_BLOCKS * PRESCAN_THREADS * 3;
        for (uint i = 0; i < to; i++) {
            if (local_list[i] != 7) {
                printf("7: %i ", local_list[i]);
            }
        }
        from = to;
        to += PRESCAN_THREADS;
        for (uint i = from; i < to; i++) {
            if (local_list[i] != 8) {
                printf("8: %i ", local_list[i]);
            }
        }
        from = to;
        to += PRESCAN_BLOCKS * PRESCAN_THREADS / 2;
        for (uint i = from; i < to; i++) {
            if (local_list[i] != 4) {
                printf("4: %i ", local_list[i]);
            }
        }
        if(local_list[to] != 1)
            printf("1: %i ", local_list[to+1]);
        if (local_list[to + 1] != -1)
            printf("-1: %i ", local_list[to + 2]);
    }
    printf("\nTotal size: %u, expected: %u\n", sz, expected_size);
}


__device__ idxtype global_sum_offset;


__global__ void 
#ifdef __CUDA_ARCH__
__launch_bounds__(PRESCAN_THREADS, 2)
#endif
pss_kernel(idxtype* arr, idxtype length, PrescanHelperStruct<idxtype>* pshs) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *pshs = get_prescan_helper_struct(arr, length, (idxtype*)&global_sum_offset);
    }
    CudaGridSynchronizer::synchronize_grid();
    auto local_pshs = *pshs;
    prescan<idxtype, true>(local_pshs, length, &global_sum_offset);
    //cooperative_inclusive_prescan(local_pshs, length, &global_sum_offset);
}

__global__ void pss_check_kernel(idxtype* arr, idxtype length, PrescanHelperStruct<idxtype>* pshs) {
    free((void*)pshs->separated_sums_arr);
    free((void*)pshs->united_sums_arr);
    size_t sum = 0;
    idxtype* out_arr = pshs->out_arr;
    for (idxtype i = 0; i < length; i++) {
        sum += arr[i];
        if (sum != out_arr[i]) {
            printf("error in element at index %u: ", i);
            printf("%llu / ", sum);
            printf("%u;\n", out_arr[i]);
        }
    }
    free((void*)pshs->in_arr);
    free((void*)pshs->out_arr);
    
}

void PrefixScanSumTest() {
    Timer timer;
    std::stringstream ss;

    Printer::kernel_properties(pss_kernel); // was regs 72, now 62

    const int length = 1000000; // heavy
    //const int length = PRESCAN_THREADS * 2 * PRESCAN_BLOCKS; // main iteration only
    //const int length = PRESCAN_THREADS * 2 * PRESCAN_BLOCKS + PRESCAN_THREADS * 2; // main + one block
    //const int length = PRESCAN_THREADS * 2 * PRESCAN_BLOCKS + PRESCAN_THREADS * 2 * 2; // main + few blocks
    //const int length = PRESCAN_THREADS * 2 * PRESCAN_BLOCKS + 100; // main + part of one block
    //const int length = PRESCAN_THREADS * 2 * PRESCAN_BLOCKS + PRESCAN_THREADS * 2 + 100; // hybrid
    //const int length = PRESCAN_THREADS * 2 + 100; // one block + part of one
    //const int length = PRESCAN_THREADS * 2 * 2; // two blocks
    //const int length = 576;  // part of one
    //const int length = 11840; // few blocks + part of one



    idxtype* arr_cpu = (idxtype*)malloc(sizeof(idxtype) * length);
    for (int i = 0; i < length; i++) {
        arr_cpu[i] = i % 7 ? (i % 3 + 1) % 2 : i % 2;
        //arr_cpu[i] = 1;
        //arr_cpu[i] = i % 3;
        //arr_cpu[i] = 0;
    }
    arr_cpu[0] = PRESCAN_THREADS * PRESCAN_BLOCKS;

    idxtype* arr_gpu;
    cudaMalloc((void**)&arr_gpu, sizeof(idxtype) * length);
    cudaMemcpy(arr_gpu, arr_cpu, sizeof(idxtype) * length, cudaMemcpyHostToDevice);

    PrescanHelperStruct<idxtype>* pss;
    cudaMalloc((void**)&pss, sizeof(PrescanHelperStruct<idxtype>));

    int threads = PRESCAN_THREADS;
    int blocks = PRESCAN_BLOCKS;
    int shared_mem = threads * 2 * sizeof(idxtype);  // 512 * 2 * 4 = ~4 kb
    timer.startCUDA();
    pss_kernel << <blocks, threads,  shared_mem>>> (arr_gpu, length, pss);
    timer.stopCUDA();  // was 0.42, now 0.33 (blocks cause of regs)
    checkCudaErrors(cudaGetLastError());
    ss << "\nPrefix scan alg processed " << length << " elements\n";
    timer.printCUDA(ss.str());

    pss_check_kernel << <1, 1 >> > (arr_gpu, length, pss);
    CudaSynchronizer::synchronize_with_instance();
    checkCudaErrors(cudaGetLastError());

    cudaFree(arr_gpu);
    free(arr_cpu);
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
    /*cudaMalloc(&list, cpm::List<int>::gpu_sizeof());
    cudaMalloc(&other_list, cpm::List<int>::gpu_sizeof());*/
    cudaMalloc(&list, sizeof(cpm::List<int>));
    cudaMalloc(&other_list, sizeof(cpm::List<int>));
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
    int shared_mem = threads * 2 * sizeof(idxtype);  // 512 * 2 * 4 = ~4 kb
    checkCudaErrors(cudaLaunchCooperativeKernel((void*)list_insert_kernel, blocks, threads, args, shared_mem, nullptr));
    timer.stopCUDA();
    checkCudaErrors(cudaGetLastError());
    ss.clear();
    ss << "\nInserted " << expected_list_size << " elements to list \n";
    timer.printCUDA(ss.str());

    list_print_kernel << <1, 1 >> > (list, expected_list_size, false);
    CudaSynchronizer::synchronize_with_instance();
    checkCudaErrors(cudaGetLastError());
}