#include "Initializer.cuh"
#include "RefractiveTable.cuh"
#include "Defines.cuh"

__global__ void initialize_models_kernel(ModelConstructInfo* mci, int length, Model* out_models) {
    int ind = blockIdx.x * blockDim.x + threadIdx.x;

    if (ind >= length) return;

    out_models[ind] = Model(mci[ind], ind);
}

cpm::pair<Model*, Model*> initialize_models(std::vector<ModelConstructInfo>& objs, int length) {
    auto gpu_mci = ModelConstructInfo::vector_to_device(objs);

    Model* gpu_models;
    cudaMalloc(&gpu_models, length * (sizeof(ModelConstructInfo) + sizeof(int)));

    int threads = 256;
    int blocks = (length + threads - 1) / threads;
    initialize_models_kernel << <blocks, threads >> > (gpu_mci, length, gpu_models);
    checkCudaErrors(cudaDeviceSynchronize());

    Model* cpu_models = (Model*)malloc(length * sizeof(Model));
    for (int i = 0; i < objs.size(); i++) {
        cpu_models[i] = Model(objs[i], i);
    }

    return { cpu_models, gpu_models };
}

cpm::pair<LightSource*, LightSource*> initialize_light_sources(
    const cpm::pair<std::vector<ModelConstructInfo>, std::vector<std::string>>& mci_and_names, int* out_size) {
    int models_number = mci_and_names.first.size();
    int light_sources_number = 0;
    std::vector<LightSource> light_sources;
    for (int i = 0; i < models_number; i++) {
        if (mci_and_names.second[i] == "light") {
            cpm::vec3 light_source_position(0.f);
            int vertices_size = mci_and_names.first[i].size;
            for (int j = 0; j < vertices_size; j++) {
                light_source_position += mci_and_names.first[i].positions[j];
            }
            light_source_position /= vertices_size;
            light_sources.push_back(LightSource(light_source_position));
            light_sources_number++;
        }
    }

    LightSource* cpu_light_sources = (LightSource*)malloc(light_sources_number * sizeof(LightSource));
    for (int i = 0; i < light_sources_number; i++) {
        cpu_light_sources[i] = light_sources[i];
    }

    LightSource* gpu_light_sources;
    checkCudaErrors(cudaMalloc(&gpu_light_sources, light_sources_number * sizeof(LightSource)));
    checkCudaErrors(cudaMemcpy(gpu_light_sources, cpu_light_sources, light_sources_number * sizeof(LightSource), cudaMemcpyHostToDevice));

    *out_size = light_sources_number;
    return { cpu_light_sources, gpu_light_sources };
}

void initialize_refractive_tables(Scene& cpu_scene) {
    cpu_refractive_table = RefractiveTable(cpu_scene);
    checkCudaErrors(cudaMemcpyToSymbol(gpu_refractive_table, cpu_refractive_table.refractive_table, 
        sizeof(float) * cpu_refractive_table.table_size, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpyToSymbol(gpu_refractive_table, &cpu_refractive_table.default_refr,
        sizeof(float), (size_t)(sizeof(float) * MAX_REFRACTIVE_TABLE_SIZE), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpyToSymbol(gpu_refractive_table, &cpu_refractive_table.table_size,
        sizeof(float), (size_t)(sizeof(float) * MAX_REFRACTIVE_TABLE_SIZE + 1), cudaMemcpyHostToDevice));

    // CUDA Error 1, ????????????????????????????
    /*checkCudaErrors(cudaMemcpyToSymbol(gpu_default_refraction, &cpu_refractive_table.default_refr, sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpyToSymbol(gpu_refractive_table_size, &cpu_refractive_table.table_size, sizeof(int), cudaMemcpyHostToDevice));*/
}


cpm::pair<Scene*, Scene*> initialize_scene(const std::string& path, const std::string& filename, Camera camera) {
    auto mci_and_names = loadOBJ(path, filename);
    int models_number = mci_and_names.first.size();
    int light_sources_number = 0;
    cpm::CudaRandom cuRandom(models_number);

    // Light source initialization must be first due to the swap method for mci in the Model constructor
    cpm::pair<LightSource*, LightSource*> cpu_gpu_light_sources = initialize_light_sources(mci_and_names, &light_sources_number);
    cpm::pair<Model*, Model*> cpu_gpu_models = initialize_models(mci_and_names.first, models_number);

    Scene* cpu_scene = new Scene(cpu_gpu_models.first, models_number, cpu_gpu_light_sources.first, light_sources_number, camera);

    Scene fake_gpu_scene(cpu_gpu_models.second, models_number, cpu_gpu_light_sources.second, light_sources_number, camera);
    Scene* gpu_scene;
    cudaMalloc(&gpu_scene, sizeof(Scene));
    cudaMemcpy(gpu_scene, &fake_gpu_scene, sizeof(Scene), cudaMemcpyHostToDevice);

    initialize_refractive_tables(*cpu_scene);

    return { cpu_scene, gpu_scene };
}