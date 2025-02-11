#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "Window.cuh"
#include "Printer.cuh"
#include "Initializer.cuh"



int main(int argc, char * argv[]) {
    setlocale(LC_ALL, "");
    cudaSetDevice(0);
    int width  = 1024;
    int height = 768;
    /*int width = 2560;
    int height = 1080;*/
   /* int width = 100;
    int height = 100;*/

    Printer::cuda_properties();

    initialize_global_params(3, 1.f);

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