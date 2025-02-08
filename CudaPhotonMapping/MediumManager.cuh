#pragma once
#include <crt/host_defines.h>
#include <cuda_runtime.h>
#include "DeepLookStack.cuh"
#include "Pair.cuh"
#include "Tuple3.h"


#define MediumManager_Version_1

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

#ifdef MediumManager_Version_1
    MMOuterContainer mediums_stack;
#endif // MediumManager_Version_1
private:
public:
    MediumManager() : mediums_stack(0) {}
#ifdef MediumManager_Version_1
    MediumManager(int stack_capacity) :  mediums_stack(stack_capacity) {}
#endif // MediumManager_Version_1

    __host__ __device__ void intialize(
        int stack_capacity, float default_refractive_index, unsigned int default_inside_level) {
        
        mediums_stack = MMOuterContainer(stack_capacity);
        mediums_stack.set_size(1);
        auto mediums_data = mediums_stack.get_data();

        /* 4 (outer pointer) + 4 + 4 (size, capacity) + 8 (outer stack_capacity) * 
        *  (4 (inner pointer) + 4 + 4 (size, capacity) + 8 (inner stack_capacity) * ( 4 (refr index) + 4 (hit id))) =
        * = 116
        */
        /* 24 + 8 * (24 + 8 * 8) = 
        */
        for (int i = 0; i < stack_capacity; i++) {
            mediums_data[i] = cpm::DeepLookStack<MediumContent>(stack_capacity);
            mediums_data[i].push({default_refractive_index, -1});
        }
    }
    __host__ __device__ cpm::Tuple3<float, float, bool> get_refractive_indices(
        size_t model_id, float model_refractive_index) {
        
        auto mediums = mediums_stack.top();
        auto medium = mediums.top();

        if (medium.hit_id == model_id) {
            return { model_refractive_index, mediums.top(1).refractive_index, false };
        }

        return { medium.refractive_index, model_refractive_index, true };
    }
    __host__ __device__ void increase_depth(bool& replace_medium) {
        if (replace_medium) {
            replace_medium = false;
            return;
        }

        if (mediums_stack.get_size() < 2) {
            mediums_stack.push_copy(mediums_stack.top());   
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
        auto medium = mediums->top();
        if (medium.hit_id == model_id) {
            mediums->pop();
        }
        else {
            mediums->push({ model_refractive_index, model_id });
        }
    }
    __host__ __device__ void set_data(MMInnerContainer* medium_manager_inner_container, 
        MMInnerData* medium_manager_innder_data, size_t stack_cap, 
        float default_refractive_index) {
        mediums_stack.set_data(medium_manager_inner_container, stack_cap, 1);

        for (size_t i = 0; i < stack_cap; i++) {
            medium_manager_inner_container[i].set_data(medium_manager_innder_data + stack_cap * i, stack_cap);
            medium_manager_inner_container[i].push({ default_refractive_index, -1});
        }
    }

};