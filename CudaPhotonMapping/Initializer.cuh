#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <map>
#include "Scene.cuh"
#include "ObjLoader.h"
#include "GlobalParams.cuh"

cpm::pair<Scene*, Scene*> initialize_scene(const std::string& path, const std::string& filename, Camera camera);

void initialize_global_params(int max_depth, float default_refractive_index);