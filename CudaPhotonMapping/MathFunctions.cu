#include "MathFunctions.cuh"

__host__ __device__ int next_power_of_two(int x) {
    int power = 1;
    while (power < x) {
        power *= 2;
    }
    return power;
}