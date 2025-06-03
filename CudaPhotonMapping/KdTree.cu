#include "KdTree.cuh"
#include "Timer.cuh"
#include "Array.cuh"
#include "SoArray.cuh"
#include "MathFunctions.cuh"
#include <cuda_runtime.h>
#include "PrefixScanSum.cuh"

using Node = KdTree::Node;

struct Chunk {
	struct Data {
		idxtype node_owner_ind;
		idxtype prim_start_ind;
		int count;
		int atomic_primitive_length;
		AABB bbox;
	};
	idxtype* node_owner_ind;
	idxtype* prim_start_ind;
	int* count;
	int* atomic_primitive_length;
	AABB* bbox;

	__host__ __device__
	void add(idxtype idx, const Chunk::Data& chunk_data) {
		node_owner_ind[idx] = chunk_data.node_owner_ind;
		prim_start_ind[idx] = chunk_data.prim_start_ind;
		count[idx] = chunk_data.count;
		atomic_primitive_length[idx] = chunk_data.atomic_primitive_length;
		bbox[idx] = chunk_data.bbox;
	}

	__host__
	void initialize_on_device(idxtype capacity) {
		cudaMalloc(&node_owner_ind, sizeof(idxtype) * capacity);
		cudaMalloc(&prim_start_ind, sizeof(idxtype) * capacity);
		cudaMalloc(&count, sizeof(int) * capacity);
		cudaMalloc(&atomic_primitive_length, sizeof(int) * capacity);
		cudaMalloc(&bbox, sizeof(AABB) * capacity);
	}

	__host__
	void free_fields() {
		cudaFree(node_owner_ind);
		cudaFree(prim_start_ind);
		cudaFree(count);
		cudaFree(atomic_primitive_length);
		cudaFree(bbox);
	}
};

// TODO and TEST shared

//#define AABBS_PER_BLOCK 512
//
//__global__
//void calculate_bounding_boxes(AABB* aabbs, cpm::vec3* positions, idxtype size, int primitive_length) {
//	extern __shared__ cpm::vec3 shared_positions[];
//
//	int vertices_per_block = AABBS_PER_BLOCK * primitive_length;
//	idxtype block_shift = vertices_per_block * blockIdx.x;
//
//	idxtype start = threadIdx.x + block_shift;
//	idxtype end = min(block_shift + vertices_per_block, size);
//
//	for (idxtype i = start; i < end; i += blockDim.x) {
//		shared_positions[i - block_shift] = positions[i];
//	}
//
//	__syncthreads();
//
//	start = threadIdx.x + block_shift / primitive_length;
//	end = min(block_shift / primitive_length + AABBS_PER_BLOCK, size / primitive_length);
//
//	for (idxtype i = start; i < end; i += blockDim.x) {
//		aabbs[i].fill(&shared_positions[(i - block_shift / primitive_length) * primitive_length], primitive_length);
//	}
//}

// AABB Calculation
__global__
void calculate_bounding_boxes(AABB* aabbs, cpm::vec3* positions, idxtype size, int primitive_length) {
	idxtype start = threadIdx.x + blockIdx.x * blockDim.x;
	idxtype total_primitives = size / primitive_length;
	for (idxtype i = start; i < total_primitives; i += gridDim.x * blockDim.x) {
		aabbs[i].fill(&positions[i * primitive_length], primitive_length);
	}
}

__global__
void create_chunks_kernel(cpm::Array<Node> active_nodes_arr, idxtype* per_node_chunk_offsets, Chunk chunks, int primitive_length) {
	// Split primitives into fix size chunks
	idxtype start = threadIdx.x + blockIdx.x * blockDim.x;
	idxtype size = active_nodes_arr.get_size();
	Node* active_nodes = active_nodes_arr.get_data();
	for (idxtype node_ind = start; node_ind < size; node_ind += gridDim.x * blockDim.x) {
		idxtype chunk_offset = per_node_chunk_offsets[node_ind];
		KdTree::Node node = active_nodes[node_ind];

		if (node.isLeaf) { continue; }

		int node_prim_count = node.primitiveCount;
		int chunks_in_node = (node_prim_count + KD_TREE_CHUNK_SIZE - 1) / KD_TREE_CHUNK_SIZE;
		idxtype prim_start_ind = 0;
		for (idxtype chunk_ind = 0; chunk_ind < chunks_in_node; chunk_ind++) {
			int remaining = node_prim_count - prim_start_ind;
			int chunk_prim_count = min(KD_TREE_CHUNK_SIZE, remaining);
			chunks.add(chunk_offset + chunk_ind, { node_ind, prim_start_ind, chunk_prim_count, primitive_length, {} });
			prim_start_ind += chunk_prim_count;
		}
	}
}

__device__ idxtype chunks_capacity;

__global__
void compute_required_chunks_capacity_arr(cpm::Array<Node> active_nodes_arr, idxtype* per_chunk_capacities) {
	__shared__ idxtype shared[32];

	idxtype threadId = threadIdx.x + blockIdx.x * blockDim.x;
	idxtype size = active_nodes_arr.get_size();
	Node* active_nodes = active_nodes_arr.get_data();

	for (idxtype node_ind = threadId; node_ind < size; node_ind += gridDim.x * blockDim.x) {
		KdTree::Node node = active_nodes[node_ind];

		if (node.isLeaf) { continue; }

		int node_prim_count = node.primitiveCount;
		idxtype chunks_in_node = (node_prim_count + KD_TREE_CHUNK_SIZE - 1) / KD_TREE_CHUNK_SIZE;
		per_chunk_capacities[node_ind] = chunks_in_node;
	}

	if (threadId == 0) {
		chunks_capacity = 0;
	}
}

template <typename T>
__device__ __forceinline__
T warp_reduce_sum(T val) {
	for (int offset = 16; offset > 0; offset /= 2)
		val += __shfl_down_sync(0xFFFFFFFF, val, offset);
	return val;
}

__global__
void compute_required_chunks_capacity(idxtype* per_chunk_capacities, idxtype size) {
	__shared__ idxtype shared[32];

	idxtype threadId = threadIdx.x + blockIdx.x * blockDim.x;
	int lane = threadIdx.x % 32;
	int warp_id = threadIdx.x / 32;

	idxtype sum = 0;
	if (threadId < size)
		sum = per_chunk_capacities[threadId];

	sum = warp_reduce_sum(sum);

	if (lane == 0)
		shared[warp_id] = sum;

	__syncthreads();

	if (warp_id == 0) {
		idxtype val = (lane < (blockDim.x / 32)) ? shared[lane] : 0;
		val = warp_reduce_sum(val);
		if (threadIdx.x == 0)
			atomicAdd(&chunks_capacity, val);
	}
}

__global__
#ifdef __CUDA_ARCH__
__launch_bounds__(PRESCAN_THREADS, 2)
#endif
void compute_chunks_offsets(PrescanHelperStruct<idxtype> pss, idxtype size, idxtype* total_offset) {
	prescan<idxtype, false, true>(pss, size, total_offset);
}

cpm::SoArray<Chunk, Chunk::Data> create_chunks(cpm::Array<Node> active_nodes, int primitive_length) {
	idxtype nodes_size = active_nodes.get_size();
	int threads = 512;
	int blocks = (nodes_size + threads - 1) / threads;

	idxtype fake_size = nodes_size % 2 == 0 ? nodes_size : nodes_size + 1;
	idxtype* per_node_chunk_capacities;
	cudaMalloc(&per_node_chunk_capacities, sizeof(idxtype) * fake_size);
	compute_required_chunks_capacity_arr<<<blocks, threads>>>(active_nodes, per_node_chunk_capacities);
	checkCudaErrors(cudaDeviceSynchronize());

	if (fake_size != nodes_size) { // if fake_size = nodes_size + 1
		idxtype zero = 0;
		cudaMemcpy(per_node_chunk_capacities + nodes_size, &zero, sizeof(idxtype), cudaMemcpyHostToDevice);
	}

	// array reduction
	//compute_required_chunks_capacity<<<blocks, threads>>>(per_node_chunk_capacities, nodes_size);
	//checkCudaErrors(cudaDeviceSynchronize());
	//checkCudaErrors(cudaGetLastError());

	//idxtype initial_chunks_capacity = 0;
	//cudaMemcpyFromSymbol(&initial_chunks_capacity, chunks_capacity, sizeof(idxtype), 0, cudaMemcpyDeviceToHost);

	idxtype* per_node_chunk_offsets, * block_separated_sums, * block_united_sums, * total_offset;
	cudaMalloc(&per_node_chunk_offsets, sizeof(idxtype) * fake_size);
	cudaMalloc(&block_separated_sums, sizeof(idxtype) * PRESCAN_BLOCKS);
	cudaMalloc(&block_united_sums, sizeof(idxtype) * PRESCAN_BLOCKS);
	cudaMalloc(&total_offset, sizeof(idxtype));
	PrescanHelperStruct<idxtype> pss { per_node_chunk_offsets, per_node_chunk_capacities, block_separated_sums, block_united_sums };
	
	int shared_mem = PRESCAN_THREADS * 2 * sizeof(idxtype);  // 512 * 2 * 4 = ~4 kb
	compute_chunks_offsets<<<PRESCAN_BLOCKS, PRESCAN_THREADS, shared_mem>>>(pss, fake_size, total_offset);
	checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaGetLastError());

	idxtype initial_chunks_capacity = 0;
	cudaMemcpy(&initial_chunks_capacity, total_offset, sizeof(idxtype), cudaMemcpyDeviceToHost);

	cudaFree(block_separated_sums);
	cudaFree(block_united_sums);
	cudaFree(total_offset);

	cudaFree(per_node_chunk_capacities);

	cpm::SoArray<Chunk, Chunk::Data> chunks;
	chunks.initialize_on_device(initial_chunks_capacity);

	// fill array with values
	create_chunks_kernel << <blocks, threads >> > (active_nodes, per_node_chunk_offsets, chunks.get_data(), primitive_length);
	checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaGetLastError());

	cudaFree(per_node_chunk_offsets);

	chunks.set_size(initial_chunks_capacity);

	return chunks;
}

__global__
void chunk_segment_reduction(idxtype* owners, idxtype last_chunk_idx, idxtype inner_border, idxtype pow2,
							 AABB* per_chunk_bb, AABB* per_node_bb) {
	idxtype threadId = threadIdx.x + blockIdx.x * blockDim.x;
	for (idxtype i = threadId; i < inner_border; i += gridDim.x * blockDim.x) {
		const idxtype chunk0_ind = pow2 * i;
		const idxtype chunk1_ind = pow2 * i + (pow2 >> 1);
		if (chunk1_ind > last_chunk_idx) {
			continue;
		}
		idxtype chunk0_owner = owners[chunk0_ind];
		idxtype chunk1_owner = owners[chunk1_ind];

		if (chunk0_owner != chunk1_owner) {
			// Объединяем AABB для разных сегментов
			per_node_bb[chunk1_owner].atomicAdd(per_chunk_bb[chunk1_ind]);
		}
		else {
			// Редуцируем внутри одного сегмента
			per_chunk_bb[chunk0_ind] = per_chunk_bb[chunk0_ind].add_const(per_chunk_bb[chunk1_ind]);
		}
	}

	if (threadId == 0 && inner_border == 1) { // last iteration
		per_node_bb[0].add(per_chunk_bb[0]);
	}
}

__global__
void per_chunk_aabb_reduction(AABB* chunk_aabbs, int* chunks_count, idxtype* chunks_prim_start_ind,
							  idxtype chunks_size, 
							  AABB* prim_aabbs,   idxtype* association_list) {
	// Per chunk triangle AABB reduction
	idxtype threadId = threadIdx.x + blockIdx.x * blockDim.x;

	for (idxtype chunk_ind = threadId; chunk_ind < chunks_size; chunk_ind += gridDim.x * blockDim.x) {
		int chunk_count = chunks_count[chunk_ind];
		idxtype chunk_prim_start_ind = chunks_prim_start_ind[chunk_ind];
		AABB chunk_aabb;
		for (idxtype i = 0; i < chunk_count; i++) {
			idxtype index = association_list[chunk_prim_start_ind + i];
			AABB aabb = prim_aabbs[index];
			chunk_aabb.add(aabb);
		}
		chunk_aabbs[chunk_ind] = chunk_aabb;
	}
}

cpm::Array<AABB> chunks_reduction(cpm::SoArray<Chunk, Chunk::Data> chunks_arr, idxtype nodes_size, AABB* aabbs, idxtype* association_list) {
	idxtype chunks_size = chunks_arr.get_size();
	Chunk chunks = chunks_arr.get_data();
	if (chunks_size == 0) {
		printf("Error in chunks_reduction function: zero elements in chunks_arr\n");
	}

	cpm::Array<AABB> per_chunk_aabb, per_node_aabb;
	per_chunk_aabb.initialize_on_device(chunks_size);
	per_node_aabb.initialize_on_device(nodes_size);
	per_node_aabb.fill([](idxtype i) {return AABB(); });

	int threads = 512;
	int blocks = (chunks_size + threads - 1) / threads;
	per_chunk_aabb_reduction<<<blocks, threads>>>(per_chunk_aabb.get_data(), chunks.count, chunks.prim_start_ind, 
		chunks_size, aabbs, association_list);
	checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaGetLastError());

	// Segment reduction
	idxtype pow2 = 2;
	idxtype fake_size = next_power_of_two(chunks_size);
	if (fake_size == 1) { fake_size += 1; }
	idxtype outer_border = (log2(fake_size) - 1) + 1; // inclusive border
	for (idxtype d = 0; d < outer_border; ++d) {
		idxtype inner_border = (fake_size - 1) / pow2 + 1; // inclusive border
		int blocks = (inner_border + threads - 1) / threads;
		chunk_segment_reduction<<<blocks, threads>>>(chunks.node_owner_ind, chunks_size - 1, inner_border, pow2, per_chunk_aabb.get_data(), per_node_aabb.get_data());
		checkCudaErrors(cudaDeviceSynchronize());
		checkCudaErrors(cudaGetLastError());
		pow2 *= 2;
	}

	per_chunk_aabb.free_device_from_host();
	
	return per_node_aabb;
}

__global__
void cut_empty_space_kernel(Node* active_nodes, idxtype size, cpm::List<Node> next_nodes, AABB* per_node_aabb, AABB* per_node_parent_aabb) {
	//idxtype threadId = threadIdx.x + blockIdx.x * blockDim.x;

	//for (idxtype local_node_ind = threadId; local_node_ind < size; local_node_ind += gridDim.x * blockDim.x) {
	//	

	//	AABB tight_bbox = per_node_aabb[local_node_ind];
	//	AABB parent_bbox = per_node_parent_aabb[local_node_ind];
	//	for (int side = 0; side < 6; side++) { // cut empty space by each side
	//		float node_length_by_axis = tight_bbox.length_by_axis(side % 3);
	//		float empty_space = parent_bbox.length_diff_by_side(side, tight_bbox);
	//		if (empty_space > KD_TREE_EMPTY_SPACE_PERCENTAGE * node_length_by_axis) {
	//			parent_bbox.set_side_value(side, tight_bbox.get_side_value(side));
	//		}
	//	}
	//	per_node_aabb[local_node_ind] = parent_bbox;

	//	// find axis for median split
	//	int longest_axis;
	//	float longest_length = -1.f;
	//	for (int axis = 0; axis < 3; axis++) {
	//		float length_by_axis = tight_bbox.length_by_axis(axis);
	//		if (length_by_axis > longest_length) {
	//			longest_length = length_by_axis;
	//			longest_axis = axis;
	//		}
	//	}
	//	float split_pos = tight_bbox.median_by_axis(longest_axis);
	//}

	//	node.splitAxis = longest_axis;
	//	node.splitValue = split_pos;
	//	node.leftChildIdx = 2 * node.ind; // (global) index in node list
	//	node.rightChildIdx = 2 * node.ind + 1;
	//	active_nodes[local_node_ind] = node;
	//	next_nodes.set_at({ node.leftChildIdx,  0.f, 0, false, IDXTYPE_NONE_VALUE, IDXTYPE_NONE_VALUE, 0, 0, {} }, local_node_ind * 2);
	//	next_nodes.set_at({ node.rightChildIdx, 0.f, 0, false, IDXTYPE_NONE_VALUE, IDXTYPE_NONE_VALUE, 0, 0, {} }, local_node_ind * 2 + 1);
	//	//
	//	/*for (idxtype prim_ind = node.firstPrimitiveIdx; prim_ind < node.firstPrimitiveIdx + node.primitiveCount; prim_ind++) {
	//		float centroid = positions[prim_ind * 3][longest_axis]	   +
	//						 positions[prim_ind * 3 + 1][longest_axis] +
	//						 positions[prim_ind * 3 + 2][longest_axis];
	//		if()
	//	}*/
	//}
}

void cut_empty_space(cpm::Array<Node> active_nodes, cpm::List<KdTree::Node>& next_nodes, AABB* per_node_aabb) {
	idxtype size = active_nodes.get_size();

	cpm::Array<int> cut_dicrections;
	cut_dicrections.initialize_on_device(size + 1);

	
}

cpm::Array<AABB> process_large_nodes(cpm::Array<Node> active_nodes, cpm::List<KdTree::Node>& next_nodes,
						 AABB* aabbs, idxtype* association_list, int primitive_length){
	cpm::SoArray<Chunk, Chunk::Data> chunks = create_chunks(active_nodes, primitive_length);
	cpm::Array<AABB> per_node_aabb = chunks_reduction(chunks, active_nodes.get_size(), aabbs, association_list);
	chunks.free_from_device();
	return per_node_aabb;
	
}

AABB* build_gpu_tree(cpm::vec3* positions, idxtype size, int primitive_length) {
	Timer timer;

	Node* nodes;
	/*Node* active_nodes;*/
	Node* small_nodes;

	idxtype nodes_size = 0;
	idxtype next_nodes_size = 0;
	idxtype small_nodes_size = 0;
	idxtype active_nodes_size = 1;

	idxtype primitives_number = size / primitive_length;

	/* Init active nodes */
	idxtype initial_capacity = primitives_number;
	cpm::Array<Node> active_nodes;
	active_nodes.initialize_on_device(initial_capacity);
	checkCudaErrors(cudaGetLastError());
	active_nodes.add_from_host_to_device(
		{ 0, 0.f, 0, false, IDXTYPE_NONE_VALUE, IDXTYPE_NONE_VALUE, 0, primitives_number, AABB() }
	);
	checkCudaErrors(cudaGetLastError());

	cpm::Array<idxtype> association_list;
	association_list.initialize_on_device(initial_capacity); // inital_capacity == primitives_number
	checkCudaErrors(cudaGetLastError());
	association_list.fill([](idxtype ind) { return ind; });
	checkCudaErrors(cudaGetLastError());

	/* ----------------------- */

	/* Calculate AABBs Section */
	AABB* aabbs;
	size_t initial_aabbs_bytes = primitives_number * sizeof(AABB);
	cudaMalloc(&aabbs, initial_aabbs_bytes);

	int threads = 512;
	int blocks = (primitives_number + threads - 1) / threads;
	/*timer.startCUDA();*/
	calculate_bounding_boxes << <blocks, threads >> > (aabbs, positions, size, primitive_length);
	/*timer.stopCUDA();
	checkCudaErrors(cudaGetLastError());
	std::cout << "Vertices count: " << size << "; ";
	timer.printCUDA("AABB calculated: ");*/
	//CudaSynchronizer::synchronize_with_instance();
	/* ----------------------- */

	/* Filling Active List Section */
	cpm::List<Node> next_nodes;
	auto aabb = process_large_nodes(active_nodes, next_nodes, aabbs, association_list.get_data(), primitive_length);
	
	//while (active_nodes.get_size_from_device_to_host() > 0) {
	//	process_large_nodes(active_nodes, next_nodes, aabbs, association_list.get_data(), primitive_length);
	//	/*cpm::swap<KdTree::Node*>(next_nodes, active_nodes);
	//	cpm::swap<idxtype>(next_nodes_size, active_nodes_size);*/
	//}
	active_nodes.free_device_from_host();
	association_list.free_device_from_host();
	cudaFree(aabbs);

	auto aabb_pointer = aabb.get_data();
	cudaFree(aabb.get_device_pointer());
	return aabb_pointer;
}

AABB* create_kd_tree(const Model* model) {
	auto aabb_pointer = build_gpu_tree(model->mci.positions, model->mci.size, model->mci.size / model->mci.primitives_size);
	return aabb_pointer;
}