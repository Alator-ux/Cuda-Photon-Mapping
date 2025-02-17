#pragma once
#include <crt/host_defines.h>
#include <cuda_runtime.h>
#include "DeepLookStack.cuh"
#include "Pair.cuh"
#include "Tuple3.h"
#include "Defines.cuh"
#include "GlobalParams.cuh"


class MediumManager {
public:
    struct MediumContent {
        float refractive_index;
        int   hit_id;
        //unsigned int inside_level;
    };

#define MMInnerData MediumManager::MediumContent
#define MMInnerContainer cpm::stack<MMInnerData>
#define MMOuterContainer cpm::stack<MMInnerContainer>

    MMOuterContainer mediums_stack;

private:
public:
    MediumManager() : mediums_stack(0) {}

    MediumManager(int stack_capacity) :  mediums_stack(stack_capacity) {}

    __host__ __device__ void intialize(int max_depth, int max_medium_depth, unsigned int default_inside_level) {
        
        mediums_stack = MMOuterContainer(max_depth);
        auto mediums_data = mediums_stack.get_data();

        /* 4 (outer pointer) + 4 + 4 (size, capacity) + 8 (outer stack_capacity) * 
        *  (4 (inner pointer) + 4 + 4 (size, capacity) + 8 (inner stack_capacity) * ( 4 (refr index) + 4 (hit id))) =
        * = 116
        */
        /* 24 + 8 * (24 + 8 * 8) = 
        */
        for (int i = 0; i < max_depth; i++) {
            mediums_data[i] = MMInnerContainer(max_medium_depth);
        }
    }
    __host__ __device__ cpm::Tuple3<float, float, bool> get_refractive_indices(
        size_t model_id, float model_refractive_index) {
        int level_size = mediums_stack.get_size();
        if (level_size == 0) {
            return { GlobalParams::default_refractive_index(), model_refractive_index, true };
        }

        auto mediums = mediums_stack.top();
        if (mediums.get_size() == 0) {
            return { GlobalParams::default_refractive_index(), model_refractive_index, true };
        }

        auto medium = mediums.top();
        if (medium.hit_id == model_id) {
            bool in_object = mediums.get_size() > 1;
            return { 
                model_refractive_index, 
                in_object ? mediums.top(1).refractive_index :
                            GlobalParams::default_refractive_index(),
                in_object
            };
        }

        return { medium.refractive_index, model_refractive_index, true };
    }
    __host__ __device__ void increase_depth(bool& max_level) {
        auto level_size = mediums_stack.get_size();
        if (level_size == 0) {
            mediums_stack.set_size(1);
            level_size++;
        }
        if (level_size < GlobalParams::max_depth()) {
            mediums_stack.push_copy(mediums_stack.top(), (GlobalParams::max_depth() + 1) / 2);
            max_level = false;
        }
        else {
            max_level = true;
        }
    }
    __host__ __device__ void reduce_depth(bool& max_level) {
        if (!max_level) {
            mediums_stack.pop();
        }
        max_level = false;
    }
    __host__ __device__ void inform(int model_id, float model_refractive_index, bool max_level) {
        auto mediums = mediums_stack.top_pointer(max_level ? 0 : 1);
        
        auto medium = mediums->top_pointer();
        if (medium != nullptr) {
            if (medium->hit_id == model_id) {
                mediums->pop();
            }
        }
        else {
            mediums->push({ model_refractive_index, model_id });
        }
    }
    __host__ __device__ void set_data(MMInnerContainer* medium_manager_inner_container, 
        MMInnerData* medium_manager_innder_data, int max_depth, int max_medium_depth) {
        mediums_stack.set_data(medium_manager_inner_container, max_depth);

        for (size_t i = 0; i < max_depth; i++) {
            medium_manager_inner_container[i].set_data(medium_manager_innder_data + max_medium_depth * i, max_medium_depth);
        }
    }
};