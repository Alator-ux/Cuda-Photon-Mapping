#include "Initializer.cuh"
#include "RefractiveTable.cuh"
#include "Defines.cuh"
#include "CudaGridSynchronizer.cuh"
#include "KdTree.cuh"
#include "PhotonTracing.cuh"
#include "PhotonGrid.cuh"
#include "Timer.cuh"
#include "FourTuple.cuh"

__global__ void initialize_models_kernel(ModelConstructInfo* mci, int length, Model* out_models) {
    int ind = blockIdx.x * blockDim.x + threadIdx.x;

    if (ind >= length) return;
    out_models[ind] = Model(mci[ind], ind);
}

__global__ void set_aabb_models_kernel(int length, Model* out_models, AABB** aabbs) {
    int ind = blockIdx.x * blockDim.x + threadIdx.x;

    if (ind >= length) return;
    out_models[ind].bounding_box = *aabbs[ind];
}

cpm::pair<Model*, Model*> initialize_models(std::vector<ModelConstructInfo>& objs, int length) {
    auto gpu_mci = ModelConstructInfo::vector_to_device(objs);

    Model* gpu_models;
    cudaMalloc(&gpu_models, length * sizeof(Model));

    int threads = 256;
    int blocks = (length + threads - 1) / threads;
    initialize_models_kernel << <blocks, threads >> > (gpu_mci, length, gpu_models);
    checkCudaErrors(cudaDeviceSynchronize());


    Model* cpu_models = (Model*)malloc(length * sizeof(Model));
    cudaMemcpy(cpu_models, gpu_models, length * sizeof(Model), cudaMemcpyDeviceToHost);
    std::vector<AABB*> aabb_pointers;
    for (int i = 0; i < length; i++) {
        auto aabb_pointer = create_kd_tree(&cpu_models[i]); // REORGANIZE :)
        aabb_pointers.push_back(aabb_pointer);
        cpu_models[i] = Model(objs[i], i);
        cudaMemcpy(&cpu_models[i].bounding_box, aabb_pointer, sizeof(AABB), cudaMemcpyDeviceToHost);
        checkCudaErrors(cudaGetLastError());
    }

    AABB** gpu_aabb_pointers;
    cudaMalloc(&gpu_aabb_pointers, sizeof(AABB*) * aabb_pointers.size());
    cudaMemcpy(gpu_aabb_pointers, aabb_pointers.data(), sizeof(AABB*) * aabb_pointers.size(), cudaMemcpyHostToDevice);
    checkCudaErrors(cudaGetLastError());
    set_aabb_models_kernel << <blocks, threads >> > (length, gpu_models, gpu_aabb_pointers);
    checkCudaErrors(cudaDeviceSynchronize());
   
    for (int i = 0; i < length; i++) {
        cudaFree(aabb_pointers[i]);
    }

    return { cpu_models, gpu_models };
}

cpm::pair<LightSource*, LightSource*> initialize_light_sources(
    Model* cpu_models, Model* gpu_models, int models_number,
    std::vector<std::string>& names, 
    int* out_size) {
    int light_sources_number = 0;
    std::vector<LightSource> light_sources;
    std::vector<int> ls_model_ids;
    for (int i = 0; i < models_number; i++) {
        if (names[i] == "light") {
            cpm::vec3 light_source_position(0.f);
            int vertices_size = cpu_models[i].mci.size; // replace with aabb dif 
            for (int j = 0; j < vertices_size; j++) {
                light_source_position += cpu_models[i].mci.positions[j];
            }
            light_source_position /= vertices_size;
            auto new_light_source = LightSource(light_source_position);
            new_light_source.model_owner = cpu_models + i;
            ls_model_ids.push_back(i);
            light_sources.push_back(new_light_source);
            light_sources_number++;
        }
    }

    LightSource* cpu_light_sources = (LightSource*)malloc(light_sources_number * sizeof(LightSource));
    for (int i = 0; i < light_sources_number; i++) {
        cpu_light_sources[i] = light_sources[i];
        cpu_light_sources[i].model_owner = gpu_models + ls_model_ids[i];
    }

    LightSource* gpu_light_sources;
    checkCudaErrors(cudaMalloc(&gpu_light_sources, light_sources_number * sizeof(LightSource)));
    checkCudaErrors(cudaMemcpy(gpu_light_sources, cpu_light_sources, light_sources_number * sizeof(LightSource), cudaMemcpyHostToDevice));

    for (int i = 0; i < light_sources_number; i++) {
        cpu_light_sources[i].model_owner = cpu_models + ls_model_ids[i];
    }

    *out_size = light_sources_number;
    return { cpu_light_sources, gpu_light_sources };
}
cpm::Photon* load(const std::string& filename, uint64_t& count) {
    std::ifstream file(filename, std::ios::binary);
  
    if (!file) {
        std::cerr << "Error opening file for reading: " << filename << std::endl;
        count = 0;
        return nullptr;
    }
    count = 0;
    // —читываем количество фотонов
    file.read(reinterpret_cast<char*>(&count), sizeof(uint64_t));
    if (!file) {
        std::cerr << "Failed to read photon count." << std::endl;
        count = 0;
        return nullptr;
    }

    cpm::Photon* photons = new cpm::Photon[count];

    // —читываем массив фотонов
    for (size_t i = 0; i < count; ++i) {
        file.read(reinterpret_cast<char*>(&photons[i].pos), sizeof(float) * 3);
        file.read(reinterpret_cast<char*>(&photons[i].power), sizeof(float) * 3);
        file.read(reinterpret_cast<char*>(&photons[i].inc_dir), sizeof(float) * 3);
    }
    if (!file) {
        std::cerr << "Failed to read photon data." << std::endl;
        delete[] photons;
        count = 0;
        return nullptr;
    }

    file.close();

    cpm::Photon* gpu_photons;
    cudaMalloc(&gpu_photons, sizeof(cpm::Photon) * count);
    cudaMemcpy(gpu_photons, photons, sizeof(cpm::Photon) * count, cudaMemcpyHostToDevice);
    delete[] photons;
    return gpu_photons;
}
cpm::four_tuple<PhotonGrid, PhotonGrid, PhotonGrid, PhotonGrid> initialize_photon_maps(Scene* cpu_scene, Scene* fake_gpu_scene, AABB scene_aabb) {
    int diffuse_num = 1000000;
    int specular_num = 100000;
    /*int diffuse_num = 3000;
    int specular_num = 1000;*/
    auto photon_containers = trace_photons(fake_gpu_scene, cpu_scene, diffuse_num, specular_num);

    cpm::PhotonArray diffuse_photon_arr;
    cpm::PhotonArray specular_photon_arr;
    cudaMemcpy(&diffuse_photon_arr, photon_containers.first, sizeof(cpm::PhotonArray), cudaMemcpyDeviceToHost);
    cudaMemcpy(&specular_photon_arr, photon_containers.second, sizeof(cpm::PhotonArray), cudaMemcpyDeviceToHost);


    Timer timer;
    timer.startCPU();
    PhotonGrid diffuse_grid(diffuse_num,  scene_aabb);
    diffuse_grid.build(diffuse_photon_arr.get_data());
    PhotonGrid specular_grid(specular_num, scene_aabb);
    specular_grid.build(specular_photon_arr.get_data());
    timer.stopCPU();
    timer.printCPU("Total photon maps construction duration: ");

    /*uint64_t count;
    auto diffuse_photons = load("global_map.txt", count);
    Timer timer;
    timer.startCPU();
    PhotonGrid diffuse_grid(count, scene_aabb);
    diffuse_grid.build(diffuse_photons);
    PhotonGrid specular_grid(0, scene_aabb);
    specular_grid.build(nullptr);
    timer.stopCPU();
    timer.printCPU("Total photon maps construction duration: ");*/


    return { diffuse_grid.copy_to_cpu(), specular_grid.copy_to_cpu(), diffuse_grid, specular_grid };
}


cpm::pair<cpm::pair<Scene*, Scene*>, cpm::four_tuple<PhotonGrid, PhotonGrid, PhotonGrid, PhotonGrid>> initialize_scene_and_photon_maps(const std::string& path, const std::string& filename, Camera camera) {
    auto mci_and_names = loadOBJ(path, filename);
    int models_number = mci_and_names.first.size();
    int light_sources_number = 0;
    cpm::CudaRandom cuRandom(models_number);

    // Light source initialization must be first due to the swap method for mci in the Model constructor
    cpm::pair<Model*, Model*> cpu_gpu_models = initialize_models(mci_and_names.first, models_number);
    cpm::pair<LightSource*, LightSource*> cpu_gpu_light_sources = initialize_light_sources(cpu_gpu_models.first, cpu_gpu_models.second, models_number, mci_and_names.second, &light_sources_number);

    Scene* cpu_scene = new Scene(cpu_gpu_models.first, models_number, cpu_gpu_light_sources.first, light_sources_number, camera);

    Scene fake_gpu_scene(cpu_gpu_models.second, models_number, cpu_gpu_light_sources.second, light_sources_number, camera);
    Scene* gpu_scene;
    cudaMalloc(&gpu_scene, sizeof(Scene));
    cudaMemcpy(gpu_scene, &fake_gpu_scene, sizeof(Scene), cudaMemcpyHostToDevice);

    AABB scene_aabb;
    for (int i = 0; i < cpu_scene->models_number; i++) {
        scene_aabb.add(cpu_scene->models[i].bounding_box);
    }

    auto photon_grids = initialize_photon_maps(cpu_scene, &fake_gpu_scene, scene_aabb);

    return { { cpu_scene, gpu_scene }, photon_grids };
}

void initialize_global_params(int max_depth, float default_refractive_index, int global_photon_num, int caustic_photon_num) {
    GlobalParams::set_max_depth(max_depth);
    GlobalParams::set_default_refractive_index(default_refractive_index);
    GlobalParams::set_photon_num(global_photon_num, caustic_photon_num);
}

void initialize_device_params(uint kernel_threads, uint kernel_blocks) {
    int num_sms, max_threads_per_sm, max_threads_per_block;
    cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, 0);
    cudaDeviceGetAttribute(&max_threads_per_sm, cudaDevAttrMaxThreadsPerMultiProcessor, 0);
    cudaDeviceGetAttribute(&max_threads_per_block, cudaDevAttrMaxThreadsPerBlock, 0);

    uint max_active_threads_cpu = num_sms * max_threads_per_sm;
    cudaMemcpyToSymbol(GlobalParams::max_active_threads, &max_active_threads_cpu, sizeof(uint));
    checkCudaErrors(cudaGetLastError());

    uint planned_threads_count = kernel_threads * kernel_blocks;
    uint max_active_blocks_cpu;
    if (planned_threads_count > max_active_threads_cpu) {
        max_active_blocks_cpu = max_active_threads_cpu / kernel_threads;
    }
    else {
        max_active_blocks_cpu = kernel_blocks;
    }
    
    cudaMemcpyToSymbol(GlobalParams::max_active_blocks, &max_active_blocks_cpu, sizeof(uint));
    checkCudaErrors(cudaGetLastError());

}