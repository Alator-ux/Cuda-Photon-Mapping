#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "Printer.cuh"
#include "Defines.cuh"
#include "Pair.cuh"

namespace cpm {
    class CAHashMap {
    public:
        static constexpr idxtype kHashTableCapacity = 128 * 1024;
        static constexpr uint kNumKeyValues = kHashTableCapacity / 2;
        static constexpr uint64 kEmptyPacked = 0xffffffffffffffffull;

        struct KeyValue {
            uint64 key_packed;
            float value;
        };

        KeyValue* data;

        __host__ __device__
        CAHashMap(KeyValue* data) : data(data) {}

        __host__ __device__
         CAHashMap() : data(nullptr) {}

        __host__ __device__
        idxtype hash(uint64 packed_key) {
            return (packed_key * 2654435761u) & (kHashTableCapacity - 1);
        }

        __host__
        void insert(const cpm::pair<float, float>& key, float value) {
            uint64 packed = cpm::to_uint64(key);
            idxtype slot = hash(packed);
            while (true) {
                //uint64 prev = atomicCAS(reinterpret_cast<unsigned long long*>(&data[slot].key_packed), kEmptyPacked, packed);
                uint64 prev = data[slot].key_packed;
                if (prev == kEmptyPacked) {
                    data[slot].key_packed = packed;
                    data[slot].value = value;
                    return;
                }
                else if (prev == packed) {
                    data[slot].value = value;
                    return;
                }
                slot = (slot + 1) & (kHashTableCapacity - 1);
            }
        }

        __host__ __device__
        bool lookup(const cpm::pair<float, float>& key, float& out_value) {
            uint64 packed = cpm::to_uint64(key);
            idxtype slot = hash(packed);
            while (true) {
                uint64 curr = data[slot].key_packed;
                if (curr == packed) {
                    out_value = data[slot].value;
                    return true;
                }
                if (curr == kEmptyPacked) {
                    return false;
                }
                slot = (slot + 1) & (kHashTableCapacity - 1);
            }
        }

        __host__ __device__
        bool contains(const cpm::pair<float, float>& key) {
            uint64 packed = cpm::to_uint64(key);
            idxtype slot = hash(packed);
            while (true) {
                uint64 curr = data[slot].key_packed;
                if (curr == packed) {
                    return true;
                }
                if (curr == kEmptyPacked) {
                    return false;
                }
                slot = (slot + 1) & (kHashTableCapacity - 1);
            }
        }
    };
};


