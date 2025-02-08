#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <map>
#include "Scene.cuh"
#include "ObjLoader.h"


cpm::pair<Scene*, Scene*> initialize_scene(const std::string& path, const std::string& filename, Camera camera);