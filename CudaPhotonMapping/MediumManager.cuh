#pragma once
#include <crt/host_defines.h>
#include <cuda_runtime.h>
#include "DeepLookStack.cuh"
#include "Pair.cuh"
#include "Tuple3.h"
#include "Defines.cuh"
#include "GlobalParams.cuh"
#include "MediumContent.cuh"
#include "MMOuterStack.cuh"

class MediumManager {
    MMOuterContainer mediums_stack;
public:
    MediumManager() : mediums_stack() {}

   __host__ __device__ cpm::Tuple3<float, float, bool> get_refractive_indices(
       idxtype array_ind, idxtype array_size,
       idxtype model_id, float model_refractive_index) {
        int level_size = mediums_stack.get_size();
        if (level_size == 0) {
            return { GlobalParams::default_refractive_index(), model_refractive_index, true };
        }

        auto mediums = mediums_stack.top(array_ind, array_size);
        if (mediums.get_size() == 0) {
            return { GlobalParams::default_refractive_index(), model_refractive_index, true };
        }

        auto medium = mediums.top(array_ind, level_size, mm_outer_capacity());
        if (medium.hit_id == model_id) {
            bool in_object = mediums.get_size() > 1;
            return { 
                model_refractive_index, 
                in_object ? mediums.top(array_ind, level_size, mm_outer_capacity(), 1).refractive_index :
                            GlobalParams::default_refractive_index(),
                in_object
            };
        }

        return { medium.refractive_index, model_refractive_index, true };
    }
    __host__ __device__ void increase_depth(idxtype array_ind, idxtype array_size, bool& max_level) {
        auto level_size = mediums_stack.get_size();
        if (level_size == 0) {
            mediums_stack.set_size(1);
            level_size++;
        }
        if (level_size < GlobalParams::max_depth()) {
            
            mediums_stack.push_copy(level_size, level_size - 1, array_ind, array_size, mediums_stack.top(array_ind, array_size), (GlobalParams::max_depth() + 1) / 2);
            max_level = false;
        }
        else {
            max_level = true;
        }
    }
    __host__ __device__ void reduce_depth(idxtype array_ind, idxtype array_size, bool& max_level) {
        if (!max_level) {
            mediums_stack.pop(array_ind, array_size);
        }
        max_level = false;
    }
    __host__ __device__ void inform(idxtype array_ind, idxtype array_size, int model_id, float model_refractive_index, bool max_level) {
        auto mediums = mediums_stack.top_pointer(array_ind, array_size, max_level ? 0 : 1);
        auto local_mediums = *mediums;
        auto level_size = local_mediums.get_size();
        auto outer_capacity = mm_outer_capacity();
        auto medium = local_mediums.top_pointer(array_ind, level_size, outer_capacity);
        if (medium != nullptr) {
            if (medium->hit_id == model_id) {
                mediums->pop(array_ind, level_size, outer_capacity);
            }
        }
        else {
            mediums->push(array_ind, level_size, outer_capacity, { model_refractive_index, model_id });
        }
        
    }
    __host__ __device__ void intialize(MMInnerContainer* medium_manager_inner_container, 
        MMInnerData* medium_manager_innder_data, int max_depth, int max_medium_depth) {
        mediums_stack.initialize();

        for (size_t i = 0; i < max_depth; i++) {
            medium_manager_inner_container[i].initialize();
        }
    }
};