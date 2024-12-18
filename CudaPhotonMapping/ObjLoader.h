#pragma once
#include <sstream>
#include <fstream>
#include <iostream>
#include <map>
#include "vec3.cuh"
#include "vec2.cuh"
#include "Material.cuh"
#include <vector>
#include "ModelConstructInfo.cuh"
#include "pair.cuh"

cpm::pair<std::vector<ModelConstructInfo>, std::vector<std::string>> loadOBJ(const std::string& path, const std::string& fname);

void fill_normals(const std::string& path, const std::string& fname);