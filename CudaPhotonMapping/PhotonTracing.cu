#include "PhotonTracing.cuh"
#include <cuda_runtime.h>
#include "RefractiveTable.cuh"
#include "GlobalParams.cuh"
#include "PhotonArray.cuh"
#include "PhotonTraceStack.cuh"
#include "Array.cuh"
#include "PathOperator.cuh"
#include <map>
#include "HashMap.cuh"
#include <unordered_set>
#include "Pair.cuh"
#include "Timer.cuh"
#include "CudaUtils.cuh"

using namespace std;

using namespace cpm;

#define PHOTON_BOUNCE_NUM 10

__device__ cudaRandomStateT* curand_states;

struct SubScene {
    Model* models;
    int models_number;
    LightSource* light_sources;
    int light_sources_number;
    SubScene(const Scene& scene) {
        models = scene.models;
        models_number = scene.models_number;
        light_sources = scene.light_sources;
        light_sources_number = scene.light_sources_number;
    }
};

cpm::pair<cpm::PhotonArray*, cpm::PhotonArray*> create_photon_arrays(idxtype diffuse_photons_num, idxtype specular_photons_num) {
    PhotonArray cpu_specular_photons, cpu_diffuse_photons;
    PhotonArray* gpu_specular_photons, * gpu_diffuse_photons;

    Photon* specular_photons_data, * diffuse_photons_data;
    cudaMalloc(&specular_photons_data, sizeof(Photon) * specular_photons_num);
    cudaMalloc(&diffuse_photons_data, sizeof(Photon) * diffuse_photons_num);

    cpu_specular_photons = PhotonArray(specular_photons_data, 0, specular_photons_num);
    cpu_diffuse_photons = PhotonArray(diffuse_photons_data, 0, diffuse_photons_num);

    cudaMalloc(&gpu_specular_photons, sizeof(PhotonArray));
    cudaMemcpy(gpu_specular_photons, &cpu_specular_photons, sizeof(PhotonArray), cudaMemcpyHostToDevice);

    cudaMalloc(&gpu_diffuse_photons, sizeof(PhotonArray));
    cudaMemcpy(gpu_diffuse_photons, &cpu_diffuse_photons, sizeof(PhotonArray), cudaMemcpyHostToDevice);

    return { gpu_diffuse_photons, gpu_specular_photons };
}

void free_photon_arrays(cpm::PhotonArray* specular_photons, cpm::PhotonArray* diffuse_photons) {
    PhotonArray cpu_specular_photons, cpu_diffuse_photons;

    cudaMemcpy(&cpu_specular_photons, specular_photons, sizeof(PhotonArray), cudaMemcpyDeviceToHost);
    cudaMemcpy(&cpu_diffuse_photons, diffuse_photons, sizeof(PhotonArray), cudaMemcpyDeviceToHost);

    cudaFree(cpu_specular_photons.get_data());
    cudaFree(cpu_diffuse_photons.get_data());

    cudaFree(specular_photons);
    cudaFree(diffuse_photons);

}

__host__ void calc_angle(std::map<std::pair<float, float>, float>& ca_table, float eta1, float eta2) {
    float eta = eta2 / eta1;
    if (eta < 1.f && ca_table.find({ eta1, eta2 }) == ca_table.end()) {
        float ca = std::cos(std::asin(eta));
        ca_table[{eta1, eta2}] = ca;
    }
}

__host__ CAHashMap* compute_critical_angles(Scene& scene)
{
    float eta1, eta2, eta, ca;
    std::map<std::pair<float, float>, float> ca_table;
    std::unordered_set<float> unique_refractive_indices;

    for (int i = 0; i < scene.models_number - 1; i++) {
        for (int j = i + 1; j < scene.models_number; j++) {
            eta1 = scene.models[i].get_material()->refr_index;
            eta2 = scene.models[j].get_material()->refr_index;
            calc_angle(ca_table, eta1, eta2); // from eta1 medium to eta2 medium
            calc_angle(ca_table, eta2, eta1); // from eta2 medium to eta1 medium
        }
    }
    for (int i = 0; i < scene.models_number; i++) {
        eta1 = scene.models[i].get_material()->refr_index;
        unique_refractive_indices.insert(eta1);
        calc_angle(ca_table, GlobalParams::default_refractive_index(), eta1); // from eta1 medium to eta2 medium
        calc_angle(ca_table, eta1, GlobalParams::default_refractive_index()); // from eta2 medium to eta1 medium
    }
    unique_refractive_indices.insert(GlobalParams::default_refractive_index());

    cpm::CAHashMap::KeyValue* data = new cpm::CAHashMap::KeyValue[cpm::CAHashMap::kHashTableCapacity];
    for (int i = 0; i < cpm::CAHashMap::kHashTableCapacity; i++) {
        data[i].key_packed = cpm::CAHashMap::kEmptyPacked;
    }

    cpm::CAHashMap ca_table_cpu(data);
    for each (auto kv in ca_table)
    {
        auto pair = cpm::pair<float, float>(kv.first.first, kv.first.second);
        ca_table_cpu.insert(pair, kv.second);
    }

    cpm::CAHashMap::KeyValue* gpu_data;
    cudaMalloc(&gpu_data, sizeof(cpm::CAHashMap::KeyValue) * cpm::CAHashMap::kHashTableCapacity);
    cudaMemcpy(gpu_data, data, sizeof(cpm::CAHashMap::KeyValue) * cpm::CAHashMap::kHashTableCapacity, cudaMemcpyHostToDevice);
    ca_table_cpu.data = gpu_data;

    cpm::CAHashMap* ca_table_gpu;
    cudaMalloc(&ca_table_gpu, sizeof(cpm::CAHashMap));
    cudaMemcpy(ca_table_gpu, &ca_table_cpu, sizeof(cpm::CAHashMap), cudaMemcpyHostToDevice);

    free(data);

    return ca_table_gpu;
};

__device__
cpm::pair<float, float> get_update_refraction_pair(float model_refractive_index, int model_id,
    idxtype threads_num, cpm::photon_trace_stack* trace_stacks) {
    idxtype threadId = threadIdx.x + blockIdx.x * blockDim.x;
    auto stack = trace_stacks[threadId];

    MManager::MediumContent top_medium;
    if (stack.get_size() != 0 && (top_medium = stack.top(threadId, threads_num)).hit_id == model_id) {
        float outer_refr_index;
        if (stack.get_size() == 1) {
            outer_refr_index = GlobalParams::default_refractive_index();
        }
        else {
            outer_refr_index = stack.top(threadId, threads_num, 1).refractive_index;
        }
        trace_stacks[threadId].pop(threadId, threads_num);
        return { top_medium.refractive_index, outer_refr_index };
    }

    trace_stacks[threadId].push(threadId, threads_num, { model_refractive_index, model_id });
    if (stack.get_size() == 0) { // stack is local copy!
        return { GlobalParams::default_refractive_index(), model_refractive_index };
    }
    return { top_medium.refractive_index, model_refractive_index };
}

__device__
cpm::pair<float, float> get_refraction_pair(float model_refractive_index, int model_id,
    idxtype threads_num, cpm::photon_trace_stack* trace_stacks) {
    idxtype threadId = threadIdx.x + blockIdx.x * blockDim.x;

    auto stack = trace_stacks[threadId];
    if (stack.get_size() == 0) {
        return { GlobalParams::default_refractive_index(), model_refractive_index };
    }
    auto top_medium = stack.top(threadId, threads_num);

    if (top_medium.hit_id == model_id) {
        float outer_refr_index;
        if (stack.get_size() == 1) {
            outer_refr_index = GlobalParams::default_refractive_index();
        }
        else {
            outer_refr_index = trace_stacks[threadId].top(threadId, threads_num, 1).refractive_index;
        }
        return { top_medium.refractive_index, outer_refr_index };
    }

    return { top_medium.refractive_index, model_refractive_index };
}

__device__ bool find_intersection(const SubScene& scene, const cpm::Ray& ray,
    cpm::vec3& out_normal, cpm::vec3& out_intersection_point,
    int& out_model_id, float& out_refr_index, float& out_model_opaque,
    cpm::vec3& out_model_specular, cpm::vec3& out_model_diffuse) {
    float intersection = 0.f;
    Model* incident_model = nullptr;

    int models_number = scene.models_number;
    Model* models = scene.models;
    for (int i = 0; i < models_number; i++) {
        Model* model = models + i;

        float temp_inter;
        cpm::vec3 tnormal;
        bool succ = model->intersection(ray, false, temp_inter, tnormal);
        if (succ && (intersection == 0.f || temp_inter < intersection)) {
            intersection = temp_inter;
            incident_model = model;
            out_normal = tnormal;
        }
    }
    if (incident_model == nullptr) {
        return false;
    }
    out_intersection_point = (ray.direction * intersection).add(ray.origin);
    if (cpm::vec3::dot(ray.direction, out_normal) > 0) { // add reverse bool?
        out_normal.mult(-1.f);
    }
    out_model_id = incident_model->get_id();
    Material mat = *incident_model->get_material();
    out_refr_index = mat.refr_index;
    out_model_opaque = mat.opaque;
    out_model_specular = mat.specular;
    out_model_diffuse = mat.diffuse;
    return true;
}

__device__ float FresnelSchlick(float cosNL, float n1, float n2) {
    float f0 = pow((n1 - n2) / (n1 + n2), 2);
    return f0 + (1.f - f0) * pow(1.f - cosNL, 5.f);
}

__device__ bool refract_destity(float cosNL, int model_id,
    float opaque, float refractive_index,
    idxtype threads_num, cpm::photon_trace_stack* trace_stacks,
    cpm::CAHashMap critical_angles) {
    if (opaque == 1.f) {
        return false;
    }

    auto cn = get_refraction_pair(refractive_index, model_id, threads_num, trace_stacks);
    float ca;
    if (critical_angles.lookup(cn, ca) && cosNL <= ca) {
        return false;
    }

    float refr_probability;
    refr_probability = 1.f - FresnelSchlick(cosNL, cn.first, cn.second);

    int threadId = threadIdx.x + blockIdx.x * blockDim.x;
    float e = curand_uniform(curand_states + threadId);
    if (e > refr_probability) { // TODO return e <= refr_probability
        return false;
    }
    return true;
}

__device__ PathType destiny(float cosNL, int model_id,
    float opaque, float refractive_index,
    const cpm::vec3& specular, const cpm::vec3& diffuse,
    const cpm::vec3& lphoton, idxtype threads_num,
    cpm::photon_trace_stack* trace_stacks,
    cpm::CAHashMap critical_angles) {
    if (refract_destity(cosNL, model_id, opaque, refractive_index, threads_num, trace_stacks, critical_angles)) {
        return PathType::refr;
    }
    float e;

    int threadId = threadIdx.x + blockIdx.x * blockDim.x;

    auto max_lp = max(max(lphoton.x, lphoton.y), lphoton.z);
    e = fmap_to_range(curand_uniform(curand_states + threadId), 0.f, 2.f); // upper bound = 2.f because max|d + s| = 2

    auto lp_d = diffuse * lphoton;
    auto max_lp_d = max(max(lp_d.x, lp_d.y), lp_d.z);
    auto pd = max_lp_d / max_lp;
    if (e <= pd) {
        return PathType::dif_refl;
    }

    auto lp_s = specular * lphoton;
    auto max_lp_s = max(max(lp_s.x, lp_s.y), lp_s.z);
    auto ps = max_lp_s / max_lp;
    if (e <= pd + ps) {
        return PathType::spec_refl;
    }

    return PathType::absorption;
}

__device__ bool trace(const SubScene& scene, cpm::Ray& ray, const cpm::vec3& photon_power,
    cpm::PhotonArray* specular_photons, cpm::PhotonArray* diffuse_photons,
    PathOperator* path_operators, CAHashMap critical_angles,
    idxtype threads_num, cpm::photon_trace_stack* trace_stacks) {
    idxtype threadId = threadIdx.x + blockIdx.x * blockDim.x;

    cpm::vec3 normal, inter_p;
    int model_id;
    float model_refr_index;
    cpm::vec3 model_specular;
    cpm::vec3 model_diffuse;
    float model_opaque;

    if (!find_intersection(scene, ray, normal, inter_p, model_id, model_refr_index, model_opaque, model_specular, model_diffuse)) {
        return;
    }
    Ray new_ray;
    float cosNL = cpm::vec3::dot(-ray.direction, normal);
    PathType dest = destiny(cosNL, model_id, model_opaque, model_refr_index, model_specular, model_diffuse, photon_power,
        threads_num, trace_stacks, critical_angles);

    constexpr cpm::vec3 ones_vec3(1.f);

    if (dest == PathType::refr) {
        auto cn = get_update_refraction_pair(model_refr_index, model_id, threads_num, trace_stacks);
        bool succ = ray.refract(inter_p, normal, cn.first, cn.second, new_ray);
        if (!succ) {
            printf("Unexcepcted error in Photon Tracing Step");
        }
    }
    else if (dest == PathType::dif_refl || dest == PathType::absorption && !model_specular.equal(ones_vec3)) {
        if (path_operators[threadId].response())
            specular_photons->add(Photon(inter_p, photon_power, ray.direction));
        else
            diffuse_photons->add(Photon(inter_p, photon_power, ray.direction));
    }
    else if (dest == PathType::spec_refl) {
        new_ray = ray.reflect(inter_p, normal);
    }

    if (dest == PathType::dif_refl) {
        auto local_state = curand_states[threadId];
        new_ray = ray.reflect_spherical(inter_p, normal, &local_state);
        curand_states[threadId] = local_state;
    }
    ray = new_ray;
    path_operators[threadId].inform(dest);
    return dest == PathType::absorption;
}

__global__ void emit(SubScene scene, CAHashMap critical_angles,
    cpm::PhotonArray* specular_photons, cpm::PhotonArray* diffuse_photons,
    PathOperator* path_operators,
    cpm::photon_trace_stack* trace_stacks, cudaRandomStateT* gen_states,
    idxtype threads_num) {
    auto ls = scene.light_sources[0];

    idxtype threadId = threadIdx.x + blockIdx.x * blockDim.x;
    if (threadId >= threads_num) {
        return;
    }
    idxtype diffuse_photons_capacity = diffuse_photons->get_capacity();
    Ray ray;
    bool generate_new_ray = true;
    int cur_bounce_num;
    cpm::vec3 pp;
    while (specular_photons->is_not_full() || diffuse_photons->is_not_full()) {
        if (generate_new_ray) {
            cur_bounce_num = 0;
            path_operators[threadId].clear();
            trace_stacks[threadId].set_size(0);

            cudaRandomStateT local_state = gen_states[threadId];
            auto rand_point_on_light_source = ls.model_owner->gpu_get_random_point_with_normal(local_state);
            ray.origin = rand_point_on_light_source.first + rand_point_on_light_source.second * 0.0001f;

            float prob;
            float dot;
            do {
                ray.direction.x = fmap_to_range(curand_uniform(&local_state), -1.f, 1.f);
                ray.direction.y = fmap_to_range(curand_uniform(&local_state), -1.f, 1.f);
                ray.direction.z = fmap_to_range(curand_uniform(&local_state), -1.f, 1.f);
                ray.direction.normalize();
                prob = curand_uniform(&local_state);
                dot = cpm::vec3::dot(ray.direction, rand_point_on_light_source.second);
                if (dot < 0) {
                    dot = -dot;
                    ray.direction *= -1;
                }
            } while (prob > dot);
            gen_states[threadId] = local_state;
            pp = ls.intensity / (float)diffuse_photons_capacity; // photon power
        }
        generate_new_ray = trace(scene, ray, pp, specular_photons, diffuse_photons, path_operators, critical_angles, threads_num, trace_stacks);
        cur_bounce_num++;
        generate_new_ray = generate_new_ray || (cur_bounce_num > PHOTON_BOUNCE_NUM - 1);
    }
}



cpm::pair<cpm::PhotonArray*, cpm::PhotonArray*> trace_photons(Scene* gpu_scene, Scene* cpu_scene, idxtype diffuse_num, idxtype specular_num) {
    auto rt = compute_critical_angles(*cpu_scene);
    auto photon_maps = create_photon_arrays(diffuse_num, specular_num);

    int threads = 512;
    int total_photons_num = specular_num + diffuse_num;
    int blocks = (total_photons_num + threads - 1) / threads;
    blocks = std::min(blocks, calculate_max_blocks_number(threads, emit));
    int total_threads = threads * blocks;
    auto cuda_rand = CudaRandom(total_threads);

    PathOperator* cpu_path_op = new PathOperator[total_threads];
    PathOperator* gpu_path_op;
    cudaMalloc(&gpu_path_op, sizeof(PathOperator) * total_threads);
    cudaMemcpy(gpu_path_op, cpu_path_op, sizeof(PathOperator) * total_threads, cudaMemcpyHostToDevice);
    free(cpu_path_op);

    MMInnerData* photon_trace_stack_data;
    cudaMalloc(&photon_trace_stack_data, sizeof(MMInnerData) * total_threads * PHOTON_BOUNCE_NUM);
    set_pt_stack_parameters(nullptr, photon_trace_stack_data, PHOTON_BOUNCE_NUM);
    cpm::Array<cpm::photon_trace_stack> trace_stacks;
    trace_stacks.initialize_on_device(total_threads);
    trace_stacks.fill([](idxtype i) {return cpm::photon_trace_stack(); }); // set zero size

    auto gpu_subscene = SubScene(*gpu_scene);
    CAHashMap fake_ca_table;
    cudaMemcpy(&fake_ca_table, rt, sizeof(cpm::CAHashMap), cudaMemcpyDeviceToHost);
    cudaMemcpyToSymbol(curand_states, &cuda_rand.states, sizeof(cudaRandomStateT*));

    Timer timer;
    timer.startCUDA();
    emit << <blocks, threads >> > (gpu_subscene, fake_ca_table, photon_maps.second, photon_maps.first,
        gpu_path_op, trace_stacks.get_data(), cuda_rand.states, total_photons_num);
    timer.stopCUDA();
    checkCudaErrors(cudaGetLastError());
    timer.printCUDA("Photon tracing duration: ");
    /*checkCudaErrors(cudaDeviceSynchronize());*/

    cudaFree(photon_trace_stack_data);
    cudaFree(gpu_path_op);
    trace_stacks.free_device_from_host();

    return photon_maps;
}