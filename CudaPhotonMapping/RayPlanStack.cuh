#pragma once
#include <crt/host_defines.h>
#include <cuda_runtime.h>
#include "Ray.cuh"
#include "RayPlan.cuh"
#include "Defines.cuh"
#include "Printer.cuh"
#include "GlobalParams.cuh"

#define RP_STACK_OFFSET(array_index, array_capacity, struct_size) ((array_capacity) * (struct_size) + (array_index))

namespace cpm {
    class rayplan_stack {
    protected:
        idxtype size;

    public:
        __host__ __device__ rayplan_stack() : size(0) {}

        __host__ __device__ bool isEmpty() const {
            return size == 0;
        }

        __host__ __device__ bool isFull(idxtype capacity) const {
            return capacity == 0 || size > capacity - 1;
        }
        __host__ __device__ void initialize(RayPlan*& data, idxtype& capacity, idxtype new_capacity) {
            free(data);
            data = (RayPlan*)malloc(new_capacity * sizeof(RayPlan));
            capacity = new_capacity;
        }
        __host__ __device__ void initialize() {
            size = 0;
        }
        
        __host__ __device__ void push(idxtype array_index, idxtype array_capacity, RayPlan value) {
            if (!isFull(raytrace_planner_capacity())) {
                idxtype offset = RP_STACK_OFFSET(array_index, array_capacity, size);
                raytrace_planner_data()[offset] = value;
                size += 1;
            }
            else {
                Printer::stack_error("stack is full, but tried to push");
            }
        }
        __host__ __device__ RayPlan pop(idxtype array_index, idxtype array_capacity) {
            if (size > 0) {
                idxtype offset = RP_STACK_OFFSET(array_index, array_capacity, size - 1);
                size -= 1;
                return raytrace_planner_data()[offset];
            }
        }

        __host__ __device__ RayPlan* top_pointer(idxtype array_index, idxtype array_capacity, idxtype offset = 0) const {
            if (isEmpty()) {
                return nullptr;
            }
            if (size < offset + 1) // size - 1 - offset < 0
            {
                Printer::stack_error("offset more then size, but tried to peek");
                return nullptr;
            }

            return raytrace_planner_data() + RP_STACK_OFFSET(array_index, array_capacity, size - 1 - offset);
        }
        __host__ __device__ RayPlan top(idxtype array_index, idxtype array_capacity, idxtype offset = 0) const {
            if (this->isEmpty())
            {
                Printer::stack_error("stack is empty, but tried to peek");
                return RayPlan();
            }
            if (size < offset + 1) // size - 1 - offset < 0
            {
                Printer::stack_error("offset more then size, but tried to peek");
                return RayPlan();
            }

            return raytrace_planner_data()[RP_STACK_OFFSET(array_index, array_capacity, size - 1 - offset)];
        }
        __host__ __device__ size_t get_size() const {
            return size;
        }
        __host__ __device__ void set_size(int size) {
            if (size >= 0) {
                this->size = size;
            }
        }
    };
}