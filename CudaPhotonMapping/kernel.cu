#include "cuda_runtime.h"
#include "device_launch_parameters.h"
//#include "Test.cuh"
#include "Window.cuh"
#include "ObjLoader.h"
#include "CudaRandom.cuh"
#include "Model.cuh"
#include "Printer.cuh"

__global__ void kernel(ModelConstructInfo* mci, int size, curandState* curandStates) {
    int ind = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (ind >= size) return;

    Printer printer;
    printer.s("Position[").i(ind).s("] = ").v3(mci->positions[ind]).nl();

   curandState local_state = curandStates[ind];
   auto a = cpm::fmap_to_range(1.f, 2.f, 3.f);
   printf("Default and ranged random generated floats: %f, %f\n",
        curand_uniform(&local_state), curand_uniform(&local_state));
    curandStates[ind] = local_state;
}

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

__global__ void initialize_scene_kernel() {

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

    return { cpu_scene, gpu_scene };
}

int main(int argc, char * argv[]) {
    setlocale(LC_ALL, "");
    int width  = 1024;
    int height = 768;
    /*int width = 2560;
    int height = 1080;*/
    /*int width = 100;
    int height = 100;*/

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    Printer::cuda_properties(prop);

    Camera camera(cpm::vec3(-0.00999999046, 0.795000017, 2.35000001), cpm::vec3(0.f, 0.f, -1.f), cpm::vec3(0.f, 1.f,0.f),
                  60, width, height); // TODO remove hardcode;
    cpm::pair<Scene*, Scene*> cpu_gpu_scenes = initialize_scene("./Models/cornell_box_sphere", "CornellBox-Sphere.obj", camera);

    
    Window window = Window(width, height, "Cuda Photon Mapping");
    
    window.set_scenes(cpu_gpu_scenes.first, cpu_gpu_scenes.second);
    window.Update();
    while (true) {
        window.Update();
    }
}