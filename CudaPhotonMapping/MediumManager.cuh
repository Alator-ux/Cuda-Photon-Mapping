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

    __host__ __device__ void intialize(int stack_capacity, unsigned int default_inside_level) {
        
        mediums_stack = MMOuterContainer(stack_capacity);
        auto mediums_data = mediums_stack.get_data();

        /* 4 (outer pointer) + 4 + 4 (size, capacity) + 8 (outer stack_capacity) * 
        *  (4 (inner pointer) + 4 + 4 (size, capacity) + 8 (inner stack_capacity) * ( 4 (refr index) + 4 (hit id))) =
        * = 116
        */
        /* 24 + 8 * (24 + 8 * 8) = 
        */
        for (int i = 0; i < stack_capacity; i++) {
            mediums_data[i] = cpm::DeepLookStack<MediumContent>(stack_capacity);
        }
    }
    __host__ __device__ cpm::Tuple3<float, float, bool> get_refractive_indices(
        size_t model_id, float model_refractive_index) {
        size_t mediums_size = mediums_stack.get_size();
        if (mediums_size == 0) {
            return { GlobalParams::default_refractive_index(), model_refractive_index, true };
        }
        auto mediums = mediums_stack.top();
        auto medium = mediums.top();

        if (medium.hit_id == model_id) {
            return { 
                model_refractive_index, 
                mediums_size == 1 ? GlobalParams::default_refractive_index() :
                                    mediums.top(1).refractive_index,
                false 
            };
        }

        return { medium.refractive_index, model_refractive_index, true };
    }
    __host__ __device__ void increase_depth(bool& replace_medium) {
        if (replace_medium) {
            replace_medium = false;
            return;
        }

        if (mediums_stack.get_size() == 0) {
            mediums_stack.set_size(1);
        }
        else {
            mediums_stack.push_copy(mediums_stack.top(), 1);
        }
    }
    __host__ __device__ void reduce_depth() {
        mediums_stack.pop();
    }
    __host__ __device__ void inform(int model_id, float model_refractive_index) {
        auto mediums = mediums_stack.top_pointer();
        
        if (mediums->get_size() > 0) {
            auto medium = mediums->top();
            if (medium.hit_id == model_id) {
                mediums->pop();
            }
        }
        else {
            mediums->push({ model_refractive_index, model_id });
        }
    }
    __host__ __device__ void set_data(MMInnerContainer* medium_manager_inner_container, 
        MMInnerData* medium_manager_innder_data, size_t stack_cap) {
        mediums_stack.set_data(medium_manager_inner_container, stack_cap);

        for (size_t i = 0; i < stack_cap; i++) {
            medium_manager_inner_container[i].set_data(medium_manager_innder_data + stack_cap * i, stack_cap);
        }
    }

};