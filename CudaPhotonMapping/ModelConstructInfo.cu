#include "ModelConstructInfo.cuh"
#include "CudaUtils.cuh"

int model_type_to_primitive_size(ModelType type){
    int size = -1;
    if ((int)type < 2) {
        size = (int)type + 3;
    }
    return size;
}

ModelConstructInfo* ModelConstructInfo::vector_to_device(const std::vector<ModelConstructInfo>& mcis) {
    int total_size = 0;
    for (int i = 0;i < mcis.size(); i++) {
        total_size += mcis[i].size;
    }
    

    ModelConstructInfo* device_mci;
    cpm::vec3* cuda_positions, *cuda_normals, *cpu_positions, *cpu_normals;
    cpm::vec2* cuda_texcoords, *cpu_texcoords;
    
    checkCudaErrors(cudaMalloc((void**)&device_mci, mcis.size() * sizeof(ModelConstructInfo)));
    for (int i = 0; i < mcis.size(); i++) {
        auto mci = mcis[i];
    
        checkCudaErrors(cudaMalloc((void**)&cuda_positions, mci.size * sizeof(cpm::vec3)));
        checkCudaErrors(cudaMemcpy(cuda_positions, mci.positions, mci.size * sizeof(cpm::vec3), cudaMemcpyHostToDevice));

        checkCudaErrors(cudaMalloc((void**)&cuda_texcoords, mci.size * sizeof(cpm::vec2)));
        checkCudaErrors(cudaMemcpy(cuda_texcoords, mci.texcoords, mci.size * sizeof(cpm::vec2), cudaMemcpyHostToDevice));

        checkCudaErrors(cudaMalloc((void**)&cuda_normals, mci.size * sizeof(cpm::vec3)));
        checkCudaErrors(cudaMemcpy(cuda_normals, mci.normals, mci.size * sizeof(cpm::vec3), cudaMemcpyHostToDevice));

        cpu_positions = mci.positions;
        cpu_texcoords = mci.texcoords;
        cpu_normals = mci.normals;
        mci.positions = cuda_positions;
        mci.texcoords = cuda_texcoords;
        mci.normals = cuda_normals;

        checkCudaErrors(cudaMemcpy(device_mci + i, &mci, sizeof(ModelConstructInfo), cudaMemcpyHostToDevice));

        mci.positions = cpu_positions;
        mci.texcoords = cpu_texcoords;
        mci.normals = cpu_normals;
    }
    
    return device_mci;
}