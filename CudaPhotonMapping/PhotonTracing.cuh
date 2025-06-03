#pragma once
#include "Defines.cuh"
#include "PhotonArray.cuh"
#include "Scene.cuh"

cpm::pair<cpm::PhotonArray*, cpm::PhotonArray*> create_photon_arrays(idxtype specular_photons_num, idxtype diffuse_photons_num);

void free_photon_arrays(cpm::PhotonArray* specular_photons, cpm::PhotonArray* diffuse_photons);

cpm::pair<cpm::PhotonArray*, cpm::PhotonArray*> trace_photons(Scene* gpu_scene, Scene* cpu_scene, idxtype specular_num, idxtype diffuse_num);