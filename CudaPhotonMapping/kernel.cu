#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "Window.cuh"
#include "Printer.cuh"
#include "Initializer.cuh"
#include "DynamicArrayTest.cuh"

int main(int argc, char * argv[]) {
    setlocale(LC_ALL, "");
    cudaSetDevice(0);
    
    Printer::cuda_properties();

    int width  = 1024;
    int height = 768;
    /*int width = 2560;
    int height = 1080;*/
    /*int width = 100;
    int height = 100;*/
    /*int width = 128;
    int height = 128;*/

    constexpr int max_trace_depth = 4;
    constexpr float default_refr_index = 1.f;
    constexpr int loc_global_photon_num = 2000;
    constexpr int loc_caustic_photon_num = 400;
    initialize_global_params(max_trace_depth, default_refr_index, loc_global_photon_num, loc_caustic_photon_num);

    Camera camera(cpm::vec3(-0.00999999046, 0.795000017, 2.35000001), cpm::vec3(0.f, 0.f, -1.f), cpm::vec3(0.f, 1.f,0.f),
                  60, width, height); // TODO remove hardcode;
    auto scenes_and_maps = initialize_scene_and_photon_maps("./Models/cornell_box_sphere", "CornellBox-Sphere.obj", camera);
    //Camera camera(cpm::vec3(2.f, 2.f, 4.f), cpm::vec3::normalize(cpm::vec3::vec3(-0.4f, -0.31f, -0.87f)), cpm::vec3(0.f, 1.f, 0.f),
    //    60, width, height); // TODO remove hardcode;
    //cpm::pair<Scene*, Scene*> cpu_gpu_scenes = initialize_scene("./Models/flagon", "flagon.obj", camera);

    
  Window window = Window(width, height, "Cuda Photon Mapping");
    
    window.set_scenes(scenes_and_maps.first.first, scenes_and_maps.first.second);
    window.set_maps(scenes_and_maps.second);
    window.Update();
    while (true) {
        window.Update();
    }
}