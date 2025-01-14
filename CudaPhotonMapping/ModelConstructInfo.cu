#include "ModelConstructInfo.cuh"
#include "CudaUtils.cuh"
#include <typeinfo>

int model_type_to_primitive_size(ModelType type){
    int size = -1;
    if ((int)type < 2) {
        size = (int)type + 3;
    }
    return size;
}

void initialize_cuda_data(size_t size_in_bytes, void* cpu_data, void** cuda_data) {
    checkCudaErrors(cudaMalloc(cuda_data, size_in_bytes));
    checkCudaErrors(cudaMemcpy(*cuda_data, cpu_data, size_in_bytes, cudaMemcpyHostToDevice));
}

void initialize_cuda_texture(size_t size_in_bytes, bool is_vec3, void* cpu_data, void** cuda_data,
    cudaTextureObject_t& texture) {
    initialize_cuda_data(size_in_bytes, cpu_data, cuda_data);

    cudaResourceDesc resource_desc = {};
    resource_desc.resType = cudaResourceTypeLinear;
    resource_desc.res.linear.devPtr = *cuda_data;
    resource_desc.res.linear.sizeInBytes = size_in_bytes;
    resource_desc.res.linear.desc = is_vec3 ? cudaCreateChannelDesc<float4>() : cudaCreateChannelDesc<float2>();

    cudaTextureDesc texture_desc = {};
    texture_desc.addressMode[0] = cudaAddressModeClamp;
    texture_desc.filterMode = cudaFilterModePoint;
    texture_desc.readMode = cudaReadModeElementType;
    texture_desc.normalizedCoords = 0;

    checkCudaErrors(cudaCreateTextureObject(&texture, &resource_desc, &texture_desc, nullptr));
}

ModelConstructInfo* ModelConstructInfo::vector_to_device(const std::vector<ModelConstructInfo>& mcis) {
    ModelConstructInfo* device_mci;
    cpm::vec3* cuda_positions, *cuda_normals, *cpu_positions, *cpu_normals;
    cpm::vec2* cuda_texcoords, *cpu_texcoords;

    checkCudaErrors(cudaMalloc((void**)&device_mci, mcis.size() * sizeof(ModelConstructInfo)));
    for (int i = 0; i < mcis.size(); i++) {
        auto mci = mcis[i];
    
        
        size_t size_in_bytes = mci.size * sizeof(cpm::vec3);
        /*initialize_cuda_texture(size_in_bytes, true, mci.positions, (void**)&cuda_positions, mci.positions_texture);
        initialize_cuda_texture(size_in_bytes, true, mci.normals, (void**)&cuda_normals, mci.normals_texture);*/
        initialize_cuda_data(size_in_bytes, mci.positions, (void**)&cuda_positions);
        initialize_cuda_data(size_in_bytes, mci.normals, (void**)&cuda_normals);

        size_in_bytes = mci.size * sizeof(cpm::vec2);
        //initialize_cuda_texture(size_in_bytes, false, mci.texcoords, (void**)&cuda_texcoords, mci.texcoords_texture);
        initialize_cuda_data(size_in_bytes, mci.texcoords, (void**)&cuda_texcoords);

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