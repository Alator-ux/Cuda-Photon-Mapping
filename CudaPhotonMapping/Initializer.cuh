#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <map>
#include "Scene.cuh"
#include "ObjLoader.h"
#include "GlobalParams.cuh"
#include "FourTuple.cuh"
#include "PhotonGrid.cuh"

cpm::pair<cpm::pair<Scene*, Scene*>, cpm::four_tuple<PhotonGrid, PhotonGrid, PhotonGrid, PhotonGrid>> initialize_scene_and_photon_maps(const std::string& path, const std::string& filename, Camera camera);

void initialize_global_params(int max_depth, float default_refractive_index, int global_photon_num, int caustic_photon_num);

void initialize_device_params(uint kernel_threads, uint kernel_blocks);