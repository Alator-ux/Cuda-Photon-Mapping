#pragma once
#include "vec3.cuh"
#include "AABB.cuh"
#include "Defines.cuh"
#include "CudaUtils.cuh"
#include "List.cuh"
#include "Model.cuh"

class KdTree {
public:
	struct Node {
		idxtype ind;

		float splitValue;  // Значение разделения
		uint splitAxis : 2;      // Ось (0=X, 1=Y, 2=Z)

		bool isLeaf;         // Флаг листа (1 = лист, 0 = внутр. узел)

		idxtype leftChildIdx;  // SIZE_MAX if not initialized
		idxtype rightChildIdx; // SIZE_MAX if not initialized 

		idxtype firstPrimitiveIdx;  // Индекс первого примитива в массиве
		idxtype primitiveCount;
		
		AABB bbox;
	};
private:
	

	Node* nodes;
	cpm::List<cpm::vec3> positions;
	cpm::List<idxtype> triangle_association_list;

	/*__host__ __device__ __forceinline__
	cpm::vec3 get_position(idxtype triangle)*/

	/*template<typename ListType>
	void resize_list(cpm::List<ListType> list) {
		list
	}*/
	
	void process_large_nodes(Node* active_nodes, idxtype size, Node* small_nodes, cpm::List<Node> next_nodes,
							 AABB* triangle_aabbs) {
		//Chunk* chunks;
//#ifdef __CUDA_ARCH__
//		__shared__ idxtype chunks_size = 0;
//#else
//		idxtype chunks_size = 0;
//#endif
		//idxtype chunks_size = 0;

		//// Split primitives into fix size chunks
		//for (idxtype node_ind = 0; node_ind < size; node_ind++) {
		//	Node node = active_nodes[node_ind];
		//	
		//	if (node.isLeaf) { continue; }
		//	int node_prim_count = node.primitiveCount;
		//	int chunks_in_node = (node_prim_count + KD_TREE_CHUNK_SIZE - 1) / KD_TREE_CHUNK_SIZE;
		//	idxtype prim_start_ind = 0;
		//	for (idxtype chunk_ind = chunks_size; chunk_ind < chunks_size + chunks_in_node; chunk_ind++) {
		//		int chunk_prim_count = node_prim_count - prim_start_ind;
		//		if (chunk_prim_count > KD_TREE_CHUNK_SIZE) {
		//			chunk_prim_count = KD_TREE_CHUNK_SIZE;
		//		}
		//		chunks[chunk_ind] = { node_ind, prim_start_ind, chunk_prim_count, 3, {} };
		//		prim_start_ind += chunk_prim_count * 3;
		//	}
		//	chunks_size += chunks_in_node;
		//}

		//// Per chunk triangle AABB reduction
		//for (idxtype chunk_ind = 0; chunk_ind < chunks_size; chunk_ind++) {
		//	Chunk chunk = chunks[chunk_ind];
		//	AABB chunk_aabb;
		//	for (idxtype i = 0; i < chunk.count; i++) {
		//		idxtype index = triangle_association_list[chunk.prim_start_ind + i];
		//		AABB triangle_aabb = triangle_aabbs[index]; // есть ли разница?
		//		chunk_aabb.add(triangle_aabb);
		//	}
		//	chunks[chunk_ind] = chunk;
		//}

		//// Per node AABBs
		//AABB* per_node_aabb;
		//chunk_segment_reduction(chunks, chunks_size, per_node_aabb);

		//for (idxtype local_node_ind = 0; local_node_ind < size; local_node_ind++) {
		//	Node node = active_nodes[local_node_ind];
		//	AABB tight_bbox = per_node_aabb[local_node_ind];
		//	if (!node.isLeaf && node.leftChildIdx == SIZE_MAX) { // if root node, initialize
		//		node.bbox = tight_bbox;
		//	}

		//	for (int side = 0; side < 6; side++) { // cut empty space by each side
		//		float node_length_by_axis = node.bbox.length_by_axis(side % 3);
		//		float empty_space = node.bbox.length_diff_by_side(side, tight_bbox);
		//		if (empty_space > KD_TREE_EMPTY_SPACE_PERCENTAGE * node_length_by_axis) {
		//			node.bbox.set_side_value(side, tight_bbox.get_side_value(side));
		//		}
		//	}

		//	// find axis for median split
		//	int longest_axis;
		//	float longest_length = -1.f;
		//	for (int axis = 0; axis < 3; axis++) {
		//		float length_by_axis = node.bbox.length_by_axis(axis);
		//		if (length_by_axis > longest_length) {
		//			longest_length = length_by_axis;
		//			longest_axis = axis;
		//		}
		//	}
		//	float split_pos = node.bbox.median_by_axis(longest_axis);

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

		//for (idxtype chunk_ind = 0; chunk_ind < chunks_size; chunk_ind++) {
		//	Chunk chunk = chunks[chunk_ind];
		//	int axis;
		//	int split_value;
		//	{
		//		Node node = active_nodes[chunk.node_owner_ind];
		//		axis = node.splitAxis;
		//		split_value = node.splitValue;
		//	}
		//	for (idxtype prim_asssoc_ind = chunk.prim_start_ind; prim_asssoc_ind < chunk.prim_start_ind + chunk.count; prim_asssoc_ind++) {
		//		idxtype prim_ind = triangle_association_list[prim_asssoc_ind];
		//		short vertices_to_left = 0;
		//		short vertices_to_right = 0;
		//		for (idxtype pos_ind = prim_ind; pos_ind < prim_ind + 3; pos_ind++) {
		//			cpm::vec3 pos = positions[pos_ind];
		//			if (pos[axis] <= split_value) {
		//				vertices_to_left++;
		//			}
		//			else {
		//				vertices_to_right++;
		//			}
		//		}
		//		if (vertices_to_left != 0 && vertices_to_right != 0) {

		//		}
		//	}
		//}
	}

	void preprocess_small_nodes(Node* small_nodes,  idxtype small_nodes_size) {

	}
	void process_small_nodes(Node* active_nodes, idxtype active_nodes_size,
								Node* small_nodes,  idxtype small_nodes_size) {

	}
	void preorder_traversal(Node* nodes) {

	}

public:
	void build_tree() {
		/*Node* nodes;
		Node* active_nodes;
		Node* small_nodes;
		Node* next_nodes;

		idxtype nodes_size = 0;
		idxtype next_nodes_size = 0;
		idxtype small_nodes_size = 0;
		idxtype active_nodes_size = 1;
		active_nodes[0] = { 0,  };

		AABB* aabbs;
		for (idxtype i = 0; i < size; i += 3) {
			aabbs[i] = AABB(positions + i, 3);
		}

		while (active_nodes_size > 0) {
			for (idxtype i = 0; i < active_nodes_size; ++i) {
				nodes[nodes_size++] = active_nodes[i];
			}
			next_nodes_size = 0;
			process_large_nodes(active_nodes, active_nodes_size, small_nodes, next_nodes);
			cpm::swap<Node*>(next_nodes, active_nodes);
			cpm::swap<idxtype>(next_nodes_size, active_nodes_size);
		}

		preprocess_small_nodes(small_nodes, small_nodes_size);
		active_nodes = small_nodes;
		active_nodes_size = small_nodes_size;

		while (active_nodes_size > 0) {
			nodes[nodes_size++] = active_nodes[active_nodes_size];
			next_nodes_size = 0;
			process_small_nodes(active_nodes, active_nodes_size, small_nodes, small_nodes_size);
			cpm::swap<Node*>(next_nodes, active_nodes);
			cpm::swap<idxtype>(next_nodes_size, active_nodes_size);
		}

		preorder_traversal(nodes);*/
	}
};

__host__ __device__
AABB* create_kd_tree(const Model* model);