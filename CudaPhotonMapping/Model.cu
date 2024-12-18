//#include "Model.cuh"
//
//#define MODEL_EPS 0.0001f
//
//__device__ int id_gen = 1;
//Model::Model(const Model& other) {
//    this->id = other.id;
//    this->mci = other.mci;
//}
//Model::Model(const ModelConstructInfo& mci, LightSource* ls) {
//    this->mci = mci;
//    this->id = id_gen+1;
//    id_gen++;
//}
//float triarea(float a, float b, float c)
//{
//    float s = (a + b + c) / 2.0;
//    return sqrtf(s * (s - a) * (s - b) * (s - c));
//}
//float calculateTriangleArea(const cpm::vec2& v0, const cpm::vec2& v1, const cpm::vec2& v2)
//{
//    return 0.5f * abs((v1.x() - v0.x()) * (v2.y() - v0.y()) - (v2.x() - v0.x()) * (v1.y() - v0.y()));
//}
//cpm::vec3 Model::barycentric_coords(const cpm::vec2& st, cpm::vec2& st0, cpm::vec2& st1, cpm::vec2& st2) const {
//    float areaABC = calculateTriangleArea(st0, st1, st2);
//
//    float areaPBC = calculateTriangleArea(st, st1, st2);
//    float areaPCA = calculateTriangleArea(st0, st, st2);
//    float areaPAB = calculateTriangleArea(st0, st1, st);
//
//    // Calculate the barycentric coordinates using the areas of the sub-triangles
//    float barycentricA = areaPBC / areaABC;
//    float barycentricB = areaPCA / areaABC;
//    float barycentricC = areaPAB / areaABC;
//
//    return cpm::vec3(barycentricA, barycentricB, barycentricC);
//}
//cpm::vec3 Model::barycentric_coords(const cpm::vec2& st, size_t ii0, size_t ii1, size_t ii2) const {
//    auto& st0 = mci.texcoords[ii0];
//    auto& st1 = mci.texcoords[ii1];
//    auto& st2 = mci.texcoords[ii2];
//
//    return barycentric_coords(st, st0, st1, st2);
//}
//bool Model::traingle_intersection(const cpm::Ray& ray, bool in_object, 
//                                  const cpm::vec3& v0, const cpm::vec3& v1, const cpm::vec3& v2, 
//                                  float& out_ray_parameter, cpm::vec3& out_uvw) const {
//    out_ray_parameter = 0.f;
//    // compute the plane's normal
//    cpm::vec3 v0v1 = v1 - v0;
//    cpm::vec3 v0v2 = v2 - v0;
//    // no need to normalize
//    cpm::vec3 N = cpm::vec3::cross(v0v1, v0v2); // N
//    float denom = cpm::vec3::dot(N, N);
//
//    // Step 1: finding P
//
//    // check if the ray and plane are parallel.
//    float NdotRayDirection = cpm::vec3::dot(N, ray.direction);
//    if (fabs(NdotRayDirection) < MODEL_EPS) // almost 0
//        return false; // they are parallel so they don't intersect! 
//
//    // compute t (equation 3)
//    out_ray_parameter = (cpm::vec3::dot(N, v0) - cpm::vec3::dot(N, ray.origin)) / NdotRayDirection;
//    // check if the triangle is behind the ray
//    if (out_ray_parameter < 0) return false; // the triangle is behind
//
//    // compute the intersection point using equation 1
//    cpm::vec3 P = ray.origin + out_ray_parameter * ray.direction;
//
//    // Step 2: inside-outside test
//    cpm::vec3 C; // vector perpendicular to triangle's plane
//
//    // edge 0
//    cpm::vec3 edge0 = v1 - v0;
//    cpm::vec3 vp0 = P - v0;
//    C = cpm::vec3::cross(edge0, vp0);
//    float w = cpm::vec3::dot(N, C);
//    if (w < 0) return false; // P is on the right side
//
//    // edge 2
//    cpm::vec3 edge2 = v0 - v2;
//    cpm::vec3 vp2 = P - v2;
//    C = cpm::vec3::cross(edge2, vp2);
//    float v = cpm::vec3::dot(N, C);
//    if (v < 0) return false; // P is on the right side;
//    w /= denom;
//    v /= denom;
//    float u = 1.f - v - w;
//    if (u <= 0) {
//        return false;
//    }
//    out_uvw[0] = u;
//    out_uvw[1] = v;
//    out_uvw[2] = w;
//    return true; // this ray hits the triangle
//}
//bool Model::intersection(const cpm::Ray& ray, bool in_object, float& intersection,
//                        size_t& ii0, size_t& ii1, size_t& ii2, 
//                        cpm::vec3& out_normal) const {
//    intersection = 0.f;
//    ii0 = ii1 = ii2 = 0;
//    cpm::vec3 uvw;
//    size_t i = 0;
//    bool intersection_found;
//    int primitive_index = 0;
//    while (primitive_index < mci.primitives_size) {
//        float possible_ray_parameter = 0.f;
//        cpm::vec3 possible_uvw;
//        if (mci.type == ModelType::Triangle) {
//            intersection_found = traingle_intersection(ray, in_object, 
//                                    mci.positions[i], mci.positions[i + 1], mci.positions[i + 2],
//                                    possible_ray_parameter, possible_uvw);
//            if (intersection_found && (intersection == 0 || possible_ray_parameter < intersection)) {
//                intersection = possible_ray_parameter;
//                uvw = possible_uvw;
//                ii0 = i;
//                ii1 = i + 1;
//                ii2 = i + 2;
//            }
//            i += 3;
//        }
//        else if (mci.type == ModelType::Quad) {
//            intersection_found = traingle_intersection(ray, in_object,
//                                    mci.positions[i], mci.positions[i + 1], mci.positions[i + 3],
//                                    possible_ray_parameter, possible_uvw);
//            if (intersection_found && (intersection == 0 || possible_ray_parameter < intersection)) {
//                intersection = possible_ray_parameter;
//                uvw = possible_uvw;
//                ii0 = i;
//                ii1 = i + 1;
//                ii2 = i + 3;
//            }
//            else {
//                possible_ray_parameter = 0.f;
//                intersection_found = traingle_intersection(ray, in_object,
//                                        mci.positions[i + 1], mci.positions[i + 2], mci.positions[i + 3],
//                                        possible_ray_parameter, possible_uvw);
//                if (intersection_found && (intersection == 0 || possible_ray_parameter < intersection)) {
//                    intersection = possible_ray_parameter;
//                    uvw = possible_uvw;
//                    ii0 = i + 1;
//                    ii1 = i + 2;
//                    ii2 = i + 3;
//                }
//            }
//            i += 4;
//        }
//        else {
//            printf("Unknown model vertex organization\n");
//        }
//        primitive_index++;
//    }
//    if (intersection == 0.f) {
//        return false;
//    }
//    auto& n0 = mci.normals[ii0];
//    auto& n1 = mci.normals[ii1];
//    auto& n2 = mci.normals[ii2];
//    out_normal = cpm::vec3::normalize(n0 * uvw.x() + n1 * uvw.y() + n2 * uvw.z());
//    return true;
//}
//cpm::vec3 Model::get_normal(size_t i) const {
//    return mci.normals[i];
//}
//void Model::get_normal(size_t ii0, size_t ii1, size_t ii2, cpm::vec3& point, cpm::vec3& normal) {
//    if (mci.smooth) {
//        auto uvw = barycentric_coords(point, ii0, ii1, ii2);
//        auto np = mci.vertices[ii0].position * uvw.x() +
//            mci.vertices[ii1].position * uvw.y() + mci.vertices[ii2].position * uvw.z;
//        normal = interpolate_uvw(mci.vertices[ii0].position, mci.vertices[ii1].position,
//            mci.vertices[ii2].position, uvw);
//        return;
//    }
//    normal = mci.vertices[ii0].normal;
//}
//bool Model::is_st_in_triangle(const cpm::vec2& st, size_t ii0, size_t ii1, size_t ii2, cpm::vec3& out_uvw) const {
//    out_uvw = barycentric_coords(st, ii0, ii1, ii2);
//    bool result = out_uvw.x() >= 0.0f && out_uvw.y() >= 0.0f && out_uvw.z() >= 0.0f;
//    out_uvw.normalize();
//    return result;
//}
//bool Model::interpolate_by_st(const cpm::vec2& st, cpm::vec3& out_position, cpm::vec3& out_normal) const {
//    cpm::vec3 possible_uvw;
//    int primitive_index = 0;
//    size_t i = 0;
//    bool is_in_triangle;
//    while (primitive_index < mci.primitives_size) {
//        if (mci.type == ModelType::Triangle) {
//            is_in_triangle = is_st_in_triangle(st, i, i + 1, i + 2, possible_uvw);
//            if (is_in_triangle) {
//                out_position = interpolate_uvw(mci.positions[i], mci.positions[i + 1], mci.positions[i + 2], possible_uvw);
//                out_normal   = interpolate_uvw(mci.normals[i], mci.normals[i + 1], mci.normals[i + 2], possible_uvw);
//                return true;
//            }
//            i += 3;
//        }
//        else if (mci.type == ModelType::Quad) {
//            is_in_triangle = is_st_in_triangle(st, i, i + 1, i + 3, possible_uvw);
//            if (is_in_triangle) {
//                out_position = interpolate_uvw(mci.positions[i], mci.positions[i + 1], mci.positions[i + 3], possible_uvw);
//                out_normal   = interpolate_uvw(mci.normals[i], mci.normals[i + 1], mci.normals[i + 3], possible_uvw);
//                return true;
//            }
//            else {
//                is_in_triangle = is_st_in_triangle(st, i + 1, i + 2, i + 3, possible_uvw);
//                if (is_in_triangle) {
//                    out_position = interpolate_uvw(mci.positions[i + 1], mci.positions[i + 2], mci.positions[i + 3], possible_uvw);
//                    out_normal   = interpolate_uvw(mci.normals[i + 1], mci.normals[i + 2], mci.normals[i + 3], possible_uvw);
//                    return true;
//                }
//            }
//            i += 4;
//        }
//        else {
//            printf("Unknown model vertex organization\n");
//        }
//        primitive_index++;
//    }
//    return false;
//}
//const Material* Model::get_material() const {
//    return &mci.material;
//}
//bool Model::equal(const Model& other) const {
//    return this->id == other.id;
//}
//bool Model::equal(size_t other_id) const {
//    return this->id == other_id;
//}
//size_t Model::get_id() const {
//    return id;
//}
//Tuple3<cpm::vec3> Model::get_bounding_box() const
//{
//    cpm::vec3 right_upper(mci.positions[0]), left_lower(mci.positions[0]), normal(0.f);
//    for (int i = 0; i < mci.size; i++) {
//        auto position = mci.positions[i];
//        for (int point_i = 0; point_i < 3; point_i++) {
//            if (position[point_i] > right_upper[point_i]) {
//                right_upper[point_i] = position[point_i];
//            }
//            if (position[point_i] < left_lower[point_i]) {
//                left_lower[point_i] = position[point_i];
//            }
//        }
//    }
//    for (int i = 0; i < mci.size; i++) {
//        normal += mci.normals[i];
//    }
//    normal /= mci.size;
//    normal.normalize();
//    return {left_lower, right_upper, normal };
//}
//cpm::pair<cpm::vec3, cpm::vec3> Model::get_random_point_with_normal(curandState* state) const {
//    curandState local_state = *state;
//
//    int start_ind, ind0, ind1, ind2;
//    if (mci.type == ModelType::Triangle) {
//
//        start_ind = CudaRandom::map_to_range(curand_uniform(&local_state), 0, (mci.size - 3) / 3);
//        start_ind *= 3;
//        ind0 = start_ind;
//        ind1 = start_ind + 1;
//        ind2 = start_ind + 2;
//    }
//    else if (mci.type == ModelType::Quad) {
//        start_ind = CudaRandom::map_to_range(curand_uniform(&local_state), 0, (mci.size - 4) / 4);
//        start_ind *= 4;
//        if (curand_uniform(&local_state) < 0.5f) {
//            ind0 = start_ind;
//            ind1 = start_ind + 1;
//            ind2 = start_ind + 3;
//        }
//        else {
//            ind0 = start_ind + 1;
//            ind1 = start_ind + 2;
//            ind2 = start_ind + 3;
//        }
//    }
//    else {
//        printf("Unknown model vertex organization\n");
//    }
//    cpm::vec3 uvw;
//    uvw[0] = curand_uniform(&local_state);
//    uvw[1] = CudaRandom::map_to_range(curand_uniform(&local_state), uvw.x(), 1.f);
//    uvw[2] = 1.f - uvw.x() - uvw.y();
//
//    cpm::vec3 point  = uvw.x() * mci.positions[ind0] + 
//                       uvw.y() * mci.positions[ind1] +
//                       uvw.z() * mci.positions[ind2];
//    cpm::vec3 normal = uvw.x() * mci.normals[ind0] +
//                       uvw.y() * mci.normals[ind1] +
//                       uvw.z() * mci.normals[ind2];
//
//    *state = local_state;
//
//    return { point, normal };
//}