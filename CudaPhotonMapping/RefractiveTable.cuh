//#pragma once
//#include "cuda_runtime.h"
//#include "device_launch_parameters.h"
//#include <map>
//#include "Scene.cuh"
//#include <unordered_set>
//#include "HashMap.cuh"
//
//#define MAX_REFRACTIVE_TABLE_SIZE 100
//
//struct RefractiveTable {
//    float default_refr = 0.f;
//    float refractive_table[100] = { 0 };
//    int table_size = 0;
//private:
//    __host__ void calc_angle(std::map<std::pair<float, float>, float> ca_table, float eta1, float eta2) {
//        float eta = eta2 / eta1;
//        if (eta < 1.f && ca_table.find({ eta1, eta2 }) == ca_table.end()) {
//            float ca = std::cos(std::asin(eta));
//            ca_table[{eta1, eta2}] = ca;
//        }
//    }
//
//    __host__ void compute_critical_angles(Scene& scene)
//    {
//        float eta1, eta2, eta, ca;
//        std::map<std::pair<float, float>, float> ca_table;
//        std::unordered_set<float> unique_refractive_indices;
//
//        for (int i = 0; i < scene.models_number - 1; i++) {
//            for (int j = i + 1; j < scene.models_number; j++) {
//                eta1 = scene.models[i].get_material()->refr_index;
//                eta2 = scene.models[j].get_material()->refr_index;
//                calc_angle(ca_table, eta1, eta2); // from eta1 medium to eta2 medium
//                calc_angle(ca_table, eta2, eta1); // from eta2 medium to eta1 medium
//            }
//        }
//        for (int i = 0; i < scene.models_number; i++) {
//            eta1 = scene.models[i].get_material()->refr_index;
//            unique_refractive_indices.insert(eta1);
//            calc_angle(ca_table, GlobalParams::default_refractive_index(), eta1); // from eta1 medium to eta2 medium
//            calc_angle(ca_table, eta1, GlobalParams::default_refractive_index()); // from eta2 medium to eta1 medium
//        }
//        unique_refractive_indices.insert(GlobalParams::default_refractive_index());
//
//        /*std::vector<float> refractive_indices(unique_refractive_indices.begin(), unique_refractive_indices.end());
//
//        int table_entry_length = refractive_indices.size();
//        table_size = table_entry_length * (table_entry_length * 2 + 1);
//        if (table_size > MAX_REFRACTIVE_TABLE_SIZE) {
//            std::cout << "Error: too many mediums" << std::endl;
//            exit(1);
//        }
//        for (int i = 0; i < table_entry_length; i++) {
//            float eta_from = refractive_indices[i];
//            int row_start = i * (table_entry_length * 2 + 1);
//            refractive_table[row_start] = eta_from;
//            for (int j = 0; j < table_entry_length; j++) {
//                float eta_to = refractive_indices[j];
//                float refractive_index = ca_table[{eta_from, eta_to}];
//                refractive_table[row_start + 1 + j * 2] = eta_to;
//                refractive_table[row_start + 1 + j * 2 + 1] = refractive_index;
//            }
//        }*/
//
//        cpm::CAHashMap::KeyValue* data = new cpm::CAHashMap::KeyValue[cpm::CAHashMap::kHashTableCapacity];
//        cpm::CAHashMap ca_table_cpu(data);
//        for each (auto kv in ca_table)
//        {
//            auto pair = cpm::pair<float, float>(kv.first.first, kv.first.second);
//            ca_table_cpu.insert(pair, kv.second);
//        }
//
//        cpm::CAHashMap::KeyValue* gpu_data;
//        cudaMalloc(&gpu_data, sizeof(cpm::CAHashMap::KeyValue) * cpm::CAHashMap::kHashTableCapacity);
//        cudaMemcpy(gpu_data, data, sizeof(cpm::CAHashMap::KeyValue) * cpm::CAHashMap::kHashTableCapacity, cudaMemcpyHostToDevice);
//
//        cpm::CAHashMap* ca_table_gpu;
//        cudaMalloc(&ca_table_gpu, sizeof(cpm::CAHashMap));
//        cudaMemcpy(ca_table_gpu, &ca_table_cpu, sizeof(cpm::CAHashMap), cudaMemcpyHostToDevice);
//        
//    };
//
//public:
//    RefractiveTable() : table_size(0), default_refr(0.f) {}
//    __host__ RefractiveTable(Scene& scene, float default_refr = 1.f) : default_refr(default_refr) {
//        compute_critical_angles(scene);
//    }
//};
