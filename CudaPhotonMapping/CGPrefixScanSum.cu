#include "CGPrefixScanSum.cuh"

__device__ idxtype cooperative_inclusive_prescan_main_part(idxtype* out_arr, idxtype* in_arr, idxtype length) {
#ifdef __CUDA_ARCH__
    __shared__ idxtype warp_sums[32];
    namespace cg = cooperative_groups;

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> tile = cg::tiled_partition<32>(block);

    idxtype val = (gid < length) ? in_arr[gid] : 0;

    idxtype scan = cg::inclusive_scan(tile, val, cg::plus<idxtype>());

    int lane = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;

    if (lane == 31)
        warp_sums[warp_id] = scan;

    __syncthreads();

    if (warp_id == 0 && lane < (blockDim.x + 31) / 32) {
        idxtype warp_sum = warp_sums[lane];
        idxtype warp_scan = cg::inclusive_scan(tile, warp_sum, cg::plus<idxtype>());
        warp_sums[lane] = warp_scan;
    }

    __syncthreads();

    if (warp_id > 0)
        scan += warp_sums[warp_id - 1];

    if (gid < length)
        out_arr[gid] = scan;

    return scan;
#endif
}

__device__ void cooperative_inclusive_prescan_big_part(idxtype* out_arr, idxtype* in_arr, idxtype* separated_sums_arr, idxtype length) {
    idxtype scan = cooperative_inclusive_prescan_main_part(out_arr, in_arr, length);
    

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid == blockDim.x - 1 || gid == length - 1)
        separated_sums_arr[blockIdx.x] = scan;
}

__device__ void cooperative_inclusive_prescan_small_part(idxtype* out_arr, idxtype* in_arr, idxtype length) {
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    int gid = bid * blockDim.x + tid;

    if (bid == 0 || gid >= length) return;

    idxtype offset = in_arr[bid - 1];
    out_arr[gid] += offset;
}

__device__ void cooperative_inclusive_prescan(PrescanHelperStruct<idxtype> pss, idxtype length, idxtype* global_sum_offset) {
    constexpr int elements_per_block = (PRESCAN_THREADS * 2);
    constexpr idxtype elements_per_iteration = elements_per_block * PRESCAN_BLOCKS;
    const idxtype total_iterations = (length + elements_per_iteration - 1) / elements_per_iteration;

    namespace cg = cooperative_groups;
    auto grid = cg::this_grid();

    for (int iteration = 0; iteration < total_iterations - 1; iteration++) {
        cooperative_inclusive_prescan_big_part(pss.out_arr, pss.in_arr, pss.separated_sums_arr, elements_per_iteration);
        printf("processed_elements %u, %u ",length, elements_per_iteration);
        grid.sync();
        cooperative_inclusive_prescan_main_part(pss.united_sums_arr, pss.separated_sums_arr, PRESCAN_BLOCKS);
        grid.sync();
        cooperative_inclusive_prescan_small_part(pss.out_arr, pss.united_sums_arr, elements_per_iteration);
        grid.sync();

        pss.out_arr += elements_per_iteration;
        pss.in_arr += elements_per_iteration;
    }

    idxtype elements_to_process;
    idxtype processed_elements = (elements_per_iteration - 1) * total_iterations;
    printf("processed_elements %u, %u, %u", processed_elements, length, elements_per_iteration);
    if (processed_elements + elements_per_iteration > length) {
        elements_to_process = length - processed_elements;
    }
    else {
        elements_to_process = elements_per_iteration;
    }

    cooperative_inclusive_prescan_big_part(pss.out_arr, pss.in_arr, pss.separated_sums_arr, elements_to_process);
    grid.sync();
    cooperative_inclusive_prescan_main_part(pss.united_sums_arr, pss.separated_sums_arr, PRESCAN_BLOCKS);
    grid.sync();
    cooperative_inclusive_prescan_small_part(pss.out_arr, pss.united_sums_arr, elements_to_process);
    grid.sync();
}