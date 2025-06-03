#include "PhotonGrid.cuh"
#include "PrescanCommon.cuh"
#include "PrefixScanSum.cuh"

struct int_vec3 {
    int x, y, z;
};


__host__ __device__ __forceinline__ int_vec3 separated_hash_position(const cpm::vec3& min_scene_pos, const cpm::vec3& pos,
    const GridSize3D& grid_size, float cell_size) {
    int_vec3 res;

    res.x = (int)((pos.x - min_scene_pos.x) / cell_size);
    res.y = (int)((pos.y - min_scene_pos.y) / cell_size);
    res.z = (int)((pos.z - min_scene_pos.z) / cell_size);

    res.x = min(max(res.x, 0), grid_size.x - 1);
    res.y = min(max(res.y, 0), grid_size.y - 1);
    res.z = min(max(res.z, 0), grid_size.z - 1);

    return res;
}

__host__ __device__ __forceinline__ uint cell_idx(int x, int y, int z, const GridSize3D& grid_size) {
    return x + y * grid_size.x + z * grid_size.x * grid_size.y;
}

__host__ __device__ __forceinline__ uint cell_idx(const int_vec3& sep_id, const GridSize3D& grid_size) {
    return sep_id.x + sep_id.y * grid_size.x + sep_id.z * grid_size.x * grid_size.y;
}

__host__ __device__ uint hash_position(const cpm::vec3& min_scene_pos, const cpm::vec3& pos,
    const GridSize3D& grid_size, float cell_size) {
    
    auto sep_id = separated_hash_position(min_scene_pos, pos, grid_size, cell_size);
    return cell_idx(sep_id, grid_size);
}


void PhotonGrid::compute_grid_and_cell_size(int desired_photons_per_cell)
{
    float size_x = scene_aabb.length_by_axis(0);
    float size_y = scene_aabb.length_by_axis(1);
    float size_z = scene_aabb.length_by_axis(2);

    int N = (num_photons + desired_photons_per_cell - 1) / desired_photons_per_cell; // минимальное количество ячеек

    float volume = size_x * size_y * size_z;

    cell_size = std::cbrt(volume / N);

    grid_size.x = static_cast<int>(std::ceil(size_x / cell_size));
    grid_size.y = static_cast<int>(std::ceil(size_y / cell_size));
    grid_size.z = static_cast<int>(std::ceil(size_z / cell_size));
}

__global__ 
void count_photons_per_cell(cpm::Photon* photons, int num_photons,
    cpm::vec3 min_scene_pos, GridSize3D grid_size, float cell_size,
    uint* cell_counts) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_photons) return;

    cpm::Photon p = photons[idx];
    uint cell_idx = hash_position(min_scene_pos, p.pos, grid_size, cell_size);
    uint count = atomicAdd(&cell_counts[cell_idx], 1);
}

__global__ 
void analyze_density_kernel(cpm::Photon* photons, int num_photons,
    cpm::vec3 min_scene_pos, GridSize3D grid_size, float cell_size,
    uint* cell_counts, uint* max_photons) {
    __shared__ uint shared_max[1];
    if (threadIdx.x == 0) shared_max[0] = 0;
    __syncthreads();

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_photons) return;

    cpm::Photon p = photons[idx];
    uint cell_idx = hash_position(min_scene_pos, p.pos, grid_size, cell_size);
    uint count = atomicAdd(&cell_counts[cell_idx], 1);
    atomicMax(&shared_max[0], count + 1);

    __syncthreads();
    if (threadIdx.x == 0) {
        atomicMax(max_photons, shared_max[0]);
    }
}

void PhotonGrid::analyze_density(int desired_photons_per_cell, cpm::Photon* photons) {
    cpm::vec3 min_scene_pos(scene_aabb.corners[0], scene_aabb.corners[1], scene_aabb.corners[2]);

    compute_grid_and_cell_size(desired_photons_per_cell);
    

    printf("Cell size: %.2f, grid size: x(%i), y(%i), z(%i)\n", cell_size, grid_size.x, grid_size.y, grid_size.z);

    int total_cells = grid_size.x * grid_size.y * grid_size.z;
    uint fake_size = total_cells % 2 == 0 ? total_cells : total_cells + 1;
    uint* d_max_photons = nullptr;
    cudaMalloc(&this->cell_sizes, fake_size * sizeof(uint));
    cudaMalloc(&d_max_photons, sizeof(uint));
    cudaMemset(this->cell_sizes, 0, fake_size * sizeof(uint));
    cudaMemset(d_max_photons, 0, sizeof(uint));

    int block_size = 256;
    int grid_block_size = (num_photons + block_size - 1) / block_size;
    analyze_density_kernel << <grid_block_size, block_size, sizeof(uint) >> > (photons, num_photons,
        min_scene_pos, grid_size, cell_size,
        this->cell_sizes, d_max_photons);
    cudaDeviceSynchronize();
    checkCudaErrors(cudaGetLastError());

    uint max_photons;
    cudaMemcpy(&max_photons, d_max_photons, sizeof(uint), cudaMemcpyDeviceToHost);

    printf("Max photons in a cell: %u, Average: %.2f\n", max_photons, ((float)num_photons) / total_cells);
    if (max_photons > 5 * desired_photons_per_cell) {
        printf("Warning: High photon concentration detected. Using smaller grid cells.\n");

        compute_grid_and_cell_size(desired_photons_per_cell / 5);

        total_cells = grid_size.x * grid_size.y * grid_size.z;
        fake_size = total_cells % 2 == 0 ? total_cells : total_cells + 1;

        cudaFree(this->cell_sizes);
        cudaMalloc(&this->cell_sizes, fake_size * sizeof(uint));
        cudaMemset(this->cell_sizes, 0, fake_size * sizeof(uint));

        count_photons_per_cell << <grid_block_size, block_size >> > (
            photons, num_photons,
            min_scene_pos, grid_size, cell_size,
            this->cell_sizes
            );
        checkCudaErrors(cudaDeviceSynchronize());
    }

    cudaFree(d_max_photons);
}


__global__
#ifdef __CUDA_ARCH__
__launch_bounds__(PRESCAN_THREADS, 2)
#endif
void prescan_per_cell_counts(PrescanHelperStruct<uint> pss, uint size, uint* total_offset) {
    prescan<uint, false>(pss, size, total_offset);
}

__global__ 
void assign_photons_to_cells(
    cpm::Photon* photons, int num_photons,
    uint* cell_start, uint* in_cell_offsets,
    GridSize3D grid_size, float cell_size,
    cpm::vec3 min_scene_pos,
    cpm::vec3* p_positions, cpm::vec3* p_directions, cpm::vec3* p_powers)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_photons) return;

    cpm::Photon p = photons[idx];

    uint cell_idx = hash_position(min_scene_pos, p.pos, grid_size, cell_size);

    uint count = atomicAdd(&in_cell_offsets[cell_idx], 1);

    uint write_idx = cell_start[cell_idx] + count;
    
    p_positions[write_idx] = p.pos;
    p_directions[write_idx] = p.inc_dir;
    p_powers[write_idx] = p.power;
}

void PhotonGrid::build_hash_grid(uint max_photons_per_cell, cpm::Photon* photons) {
    int total_cells = grid_size.x * grid_size.y * grid_size.z;
    cpm::vec3 min_scene_pos(scene_aabb.corners[0], scene_aabb.corners[1], scene_aabb.corners[2]);

    int threads = 256;
    int blocks = (num_photons + threads - 1) / threads;
    /*count_photons_per_cell << <grid_dim, block_size >> > (
        photons, num_photons, d_cell_counts,
        grid_size, cell_size, min_bound
        );*/
    
    uint * block_separated_sums, * block_united_sums, * total_offset;
    cudaMalloc(&block_separated_sums, sizeof(uint) * PRESCAN_BLOCKS);
    cudaMalloc(&block_united_sums, sizeof(uint) * PRESCAN_BLOCKS);
    cudaMalloc(&total_offset, sizeof(uint));
    cudaMalloc(&this->cell_starts, sizeof(uint) * total_cells);
    PrescanHelperStruct<uint> pss{ cell_starts, cell_sizes, block_separated_sums, block_united_sums };

    int shared_mem = PRESCAN_THREADS * 2 * sizeof(uint);
    // exclusive scan
    prescan_per_cell_counts<<<PRESCAN_BLOCKS, PRESCAN_THREADS, shared_mem>>>(pss, total_cells, total_offset);
    checkCudaErrors(cudaDeviceSynchronize());

    cudaFree(pss.separated_sums_arr);
    cudaFree(pss.united_sums_arr);

    uint* d_in_cell_offsets; // haha
    cudaMalloc(&d_in_cell_offsets, sizeof(uint) * total_cells);
    cudaMemset(d_in_cell_offsets, 0, sizeof(uint) * total_cells);
    
    cudaMalloc(&this->p_positions, sizeof(cpm::vec3) * num_photons);
    cudaMalloc(&this->p_directions, sizeof(cpm::vec3) * num_photons);
    cudaMalloc(&this->p_powers, sizeof(cpm::vec3) * num_photons);

    assign_photons_to_cells << <blocks, threads >> > (
        photons, num_photons, cell_starts, d_in_cell_offsets,
        grid_size, cell_size,
        min_scene_pos, this->p_positions, this->p_directions, this->p_powers);
    checkCudaErrors(cudaDeviceSynchronize());

    cudaFree(photons);
    cudaFree(d_in_cell_offsets);
}



struct CellAABB {
    int data[6];
    __host__ __device__ CellAABB(int ix, int iy, int iz, int radius_cells, GridSize3D grid_size) {
        data[0] = max(0, ix - radius_cells);
        data[1] = max(0, iy - radius_cells);
        data[2] = max(0, iz - radius_cells);
        data[3] = min(grid_size.x - 1, ix + radius_cells);
        data[4] = min(grid_size.y - 1, iy + radius_cells);
        data[5] = min(grid_size.z - 1, iz + radius_cells);
    }

    __host__ __device__ __forceinline__ int min_x() const { return data[0]; } 
    __host__ __device__ __forceinline__ int min_y() const { return data[1]; }
    __host__ __device__ __forceinline__ int min_z() const { return data[2]; }
    __host__ __device__ __forceinline__ int max_x() const { return data[3]; }
    __host__ __device__ __forceinline__ int max_y() const { return data[4]; }
    __host__ __device__ __forceinline__ int max_z() const { return data[5]; }
};

__host__ __device__ void PhotonGrid::find_nearests_in_cell(uint cell_idx, const cpm::vec3& location_point, 
    float radius, int num_to_find, PhotonMaxHeap& maxheap, uint array_idx, uint array_cap) {
    uint start = cell_starts[cell_idx];
    uint size = cell_sizes[cell_idx];

    for (uint i = 0; i < size; ++i) {
        uint photon_idx = start + i;
        cpm::vec3 pos = p_positions[photon_idx];
        pos -= location_point;
        float distance = pos.length();
        if (distance <= radius) {
            maxheap.push(photon_idx, distance, num_to_find, array_idx, array_cap);
        }
    }
}

__host__ __device__ bool PhotonGrid::find_nearests(const cpm::vec3& location_point, float radius, int num_to_find,
    PhotonMaxHeap& maxheap, uint array_idx, uint array_cap) {
    cpm::vec3 min_scene_pos(scene_aabb.corners[0], scene_aabb.corners[1], scene_aabb.corners[2]);
    
    maxheap.clear();
    auto grid_size = this->grid_size;
    auto sep_ids = separated_hash_position(min_scene_pos, location_point, grid_size, cell_size);
    uint actual_cell_idx = cell_idx(sep_ids, grid_size);

    int radius_cells = (int)ceilf(radius / cell_size);
    CellAABB cell_aabb(sep_ids.x, sep_ids.y, sep_ids.z, radius_cells, grid_size);
    
    find_nearests_in_cell(actual_cell_idx, location_point, radius, num_to_find, maxheap, array_idx, array_cap);

    for (int z = cell_aabb.min_z(); z <= cell_aabb.max_z(); ++z) {
        for (int y = cell_aabb.min_y(); y <= cell_aabb.max_y(); ++y) {
            for (int x = cell_aabb.min_x(); x <= cell_aabb.max_x(); ++x) {
                uint current_cell_idx = cell_idx(x, y, z, grid_size);
                if (current_cell_idx == actual_cell_idx) {
                    continue;
                }
                find_nearests_in_cell(current_cell_idx, location_point, radius, num_to_find, maxheap, array_idx, array_cap);
            }
        }
    }

    return maxheap.get_size() == num_to_find;
}