#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "Photon.cuh"
#include "PriorityQueue.cuh"
#include "vec3.cuh"
#include "Tree.cuh"
#include "PhotonMap.cuh"

namespace cpm
{
    class TestablePhotonMap : public cpm::PhotonMap {
    public:
        __host__ __device__ Photon get_root() {
            return this->tree.get_root()->value;
        }
        __host__ __device__ int has_left(int parent) {
            return this->tree.has_left(parent);
        }
        __host__ __device__ int has_right(int parent) {
            return this->tree.has_right(parent);
        }
        __host__ __device__ Photon get_left(int parent) {
            return this->tree.get_left(parent)->value;
        }
        __host__ __device__ Photon get_right(int parent) {
            return this->tree.get_right(parent)->value;
        }
        __host__ __device__ TestablePhotonMap(Type type, Photon* photons, int size) : cpm::PhotonMap(type, photons, size) {

        }
        __host__ __device__ void get_closest_to_point(const cpm::vec3& point, size_t count, cpm::vec3* out_points) {
            auto np = NearestPhotons(point, vec3(1), count);
            auto p = new Photon(cpm::vec3(point), cpm::vec3(1), cpm::vec3(0));
            auto n = new Node(p, 0);
            auto npn = cpm::PhotonMap::NPNode(n, max_distance);
            np.container->fpush(npn);
            locate_q(np);
            if (np.container->size() > count) {
                //np.container->pop();
            }
            for (int i = 0; i < np.container->size(); i++) {
                printf("sq dist %f\n", (*np.container)[i].sq_dist);
                out_points[i] = (*np.container)[i].node->value.pos;
            }
        }
    };

}