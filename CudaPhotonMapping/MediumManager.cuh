//#pragma once
//#include <crt/host_defines.h>
//#include <cuda_runtime.h>
//#include "DeepLookStack.cuh"
//#include "Pair.cuh"
//
//class MediumManager {
//    struct StackContent {
//        cpm::DeepLookStack<cpm::pair<float, size_t>> mediums;
//        bool exiting;
//        StackContent(size_t stack_depth, bool exiting = false) : mediums(stack_depth), exiting(exiting) {}
//        StackContent() : mediums(0), exiting(false) {}
//        /*StackContent(const cpm::DeepLookStack<cpm::pair<float, size_t>>& mediums, bool exiting) {
//            this->mediums = mediums;
//            this->exiting = exiting;
//        }*/
//    };
//    cpm::stack<StackContent> mediums_stack;
//    float default_refraction;
//    /// <summary>
//    /// ca_table — critical angle table.
//    /// Map with critical angles for each medium pair in the scene.
//    /// <param name="Key"> is the division of the refractive coefficients of the two media through which light passes</param>
//    /// <param name="Value"> is a critical angle for this mediums in radians</param>
//    /// </summary>
//    std::map<std::pair<float, float>, float> ca_table;
//    void calc_angle(float eta1, float eta2) {
//        float eta = eta2 / eta1;
//        if (eta < 1.f && ca_table.find({ eta1, eta2 }) == ca_table.end()) {
//            float ca = std::cos(std::asin(eta));
//            ca_table[{eta1, eta2}] = ca;
//        }
//    }
//public:
//    MediumManager(size_t stack_depth, float def_refr = 1.f) : default_refraction(def_refr), mediums_stack(stack_depth) {
//        clear();
//    }
//    MediumManager(float def_refr = 1.f) : default_refraction(def_refr), mediums_stack(0) {
//        clear();
//    }
//    void compute_critical_angles(Scene scene)
//    {
//        Model* models = scene.models; 
//        int models_number = scene.models_number;
//
//        float eta1, eta2, eta, ca;
//        for (int i = 0; i < models_number; i++) {
//            for (int j = i + 1; j < models_number; j++) {
//                eta1 = models[i].get_material()->refr_index;
//                eta2 = models[j].get_material()->refr_index;
//                calc_angle(eta1, eta2); // from eta1 medium to eta2 medium
//                calc_angle(eta2, eta1); // from eta2 medium to eta1 medium
//            }
//
//            eta1 = models[i].get_material()->refr_index;
//            calc_angle(default_refraction, eta1); // from eta1 medium to eta2 medium
//            calc_angle(eta1, default_refraction); // from eta2 medium to eta1 medium
//        }
//    };
//    bool can_refract(const std::pair<float, float>& cn, float cosNL) {
//        if (ca_table.find(cn) == ca_table.end()) {
//            return true;
//        }
//        return cosNL > ca_table[cn]; // <= ?
//    }
//    //std::pair<float, float> get_cur_new(const PMModel* model) {
//    //    std::pair<float, float> res;
//    //    auto& mediums = st_mediums.top().mediums;
//    //    auto& exiting = st_mediums.top().exiting;
//    //    if (model->equal(mediums.peek().second)) {
//    //        // Если луч столкнулся с объектом, в котором он находится, с внутренней стороны
//    //        // Надо достать внешнюю по отношению к объекту среду. На вершине стека лежит объект, 
//    //        // в котором находится луч
//    //        // текущая среда - среда объекта, по которому ударил луч, т.к. мы внутри
//    //        res.first = model->get_material()->refr_index;
//    //        res.second = mediums.peek(1).first;
//    //        exiting = true;
//    //        return res;
//    //    }
//    //    // Если луч входит в новый объект
//    //    res.first = mediums.peek().first;
//    //    res.second = model->get_material()->refr_index;
//    //    exiting = false;
//    //    return res;
//    //}
//    //void inform(bool refract_suc, const PMModel* model) {
//    //    if (!refract_suc) {
//    //        return;
//    //    }
//    //    auto& mediums = st_mediums.top().mediums;
//    //    auto& exiting = st_mediums.top().exiting;
//    //    if (exiting && model->equal(mediums.peek().second)) {
//    //        // если луч вышел из объекта, то убираем со стека текущую среду
//    //        mediums.pop();
//    //        exiting = false;
//    //    }
//    //    else { // иначе луч пересек еще один объект, добавляем текущую среду
//    //        mediums.push({ model->get_material()->refr_index, model->get_id() });
//    //    }
//    //}
//    //void increase_depth() {
//    //    st_mediums.push(st_mediums.top());
//    //}
//    //void reduce_depth() {
//    //    st_mediums.pop();
//    //}
//    //void clear() {
//    //    auto default_m = DeepLookStack<std::pair<float, size_t>>();
//    //    default_m.push({ default_refr, 0 });
//    //    st_mediums = std::stack<StackContent>();
//    //    st_mediums.push(StackContent(default_m, false));
//    //}
//};