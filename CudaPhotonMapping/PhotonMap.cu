//#include "PhotonMap.cuh"
//#include "Pair.cuh"
//#include "Stack.cuh"
//#include "FourTuple.cuh"
//#include "Sort.cuh"
//#include <cfloat>
//#include "ArrayTools.cuh"

/* ==========! PhotonMapp substruct !========== */
/* ========== Node struct begin ========== */
//__host__ __device__ cpm::PhotonMap::Node::Node(Photon* value, short plane) {
//    this->value = *value;
//    this->plane = plane;
//}
//__host__ __device__ cpm::PhotonMap::Node::~Node() {
//}
//
///* ========== Node struct end ========== */
//
//
//__host__ __device__ short cpm::PhotonMap::find_largest_plane(const cpm::vec3& bigp, const cpm::vec3& smallp)
//{
//    size_t largest_plane = 0;
//    {
//        cpm::vec3 dims(
//            fabs(bigp.x() - smallp.x()),
//            fabs(bigp.y() - smallp.y()),
//            fabs(bigp.z() - smallp.z())
//        );
//        float max = dims[0];
//        for (size_t i = 1; i < 3; i++) {
//            if (dims[i] > max) {
//                max = dims[i];
//                largest_plane = i;
//            }
//        }
//    }
//    return largest_plane;
//}
//
//__host__ __device__ void cpm::PhotonMap::update_cube(const cpm::vec3& p, cpm::vec3& bigp, cpm::vec3& smallp)
//{
//    for (size_t point_i = 0; point_i < 3; point_i++) {
//        if (p[point_i] > bigp[point_i]) {
//            bigp[point_i] = p[point_i];
//        }
//        if (p[point_i] < smallp[point_i]) {
//            smallp[point_i] = p[point_i];
//        }
//    }
//}
//
//__host__ __device__ void cpm::PhotonMap::fill_balanced(Photon* photons, size_t size)
//{
//    if (size == 0) {
//        printf("There is nothing to insert");
//        return;
//    }
//
//    printf("Filling balanced KD-tree (photon map) started");
//    depth = ceil(log2((double)size + 1.0));
//    /*
//    * нахождение "куба", захватывающего все точки и заполнение вектор указателей для работы с ними
//    * + перевод вектор точек в вектор указателей на точки для дальнейшего удобства
//    */
//    auto photons_pointers = new Photon*[size];
//    
//    cpm::vec3 bigp(photons[0].pos), smallp(photons[0].pos);
//    for (size_t i = 0; i < size; i++) {
//        photons_pointers[i] = &photons[i];
//        update_cube(photons[i].pos, bigp, smallp);
//    }
//
//    /*
//    * нахождение измерения с наибольшей длиной
//    */
//    auto largest_plane = find_largest_plane(bigp, smallp);
//
//    auto photonComp = PhotonComparator();
//
//    size_t count = 0;
//    // from, to, plane, node ind
//    auto ftpn_recur = cpm::stack<cpm::four_tuple<size_t, size_t, short, int>>(size);
//    ftpn_recur.push({ 0, size, largest_plane, 0 });
//    while (!ftpn_recur.isEmpty()) {
//        short plane;
//        size_t from, to;
//        int node_ind;
//        ftpn_recur.top().tie(&from, &to, &plane, &node_ind);
//        ftpn_recur.pop();
//        if (to - from == 0) {
//            //node = nullptr;
//            continue;
//        }
//        count++;
//
//        if (count % ((size / 10)) == 0) {
//            printf("\tPhotons inserted : " + count);
//        }
//        
//        if (to - from == 1) {
//            auto node = new Node(photons_pointers[from], plane);
//            tree.set_at(node_ind, node);
//            continue;
//        }
//
//        // TODO thurst?
//        // Сортировка для нахождения среднего. Сортируем указатели.
//        photonComp.plane = plane;
//        quick_sort(photons_pointers, from, to, photonComp);
//        //std::sort(std::next(photons_pointers.begin(), from), std::next(photons_pointers.begin(), to),
//        //    [&plane](const Photon* p1, const Photon* p2) { return p1->pos[plane] < p2->pos[plane]; });
//        size_t mid = (to - from) / 2 + from;
//        Photon* medium = photons_pointers[mid];
//
//        cpm::vec3 left_bigp(-FLT_MAX), left_smallp(FLT_MAX);
//        cpm::vec3 right_bigp(left_bigp), right_smallp(left_smallp);
//        array_foreach<Photon*>(
//            photons_pointers, 
//            from,
//            mid, 
//            [&left_bigp, &left_smallp, this](Photon* p) {
//                update_cube(p->pos, left_bigp, left_smallp);
//            }
//        );
//        array_foreach<Photon*>(
//            photons_pointers,
//            mid + 1,
//            to,
//            [&right_bigp, &right_smallp, this](const Photon* p) {
//                update_cube(p->pos, right_bigp, right_smallp);
//            }
//        );
//
//        /*std::for_each(std::next(photons_pointers.begin(), from), std::next(photons_pointers.begin(), mid),
//            [&left_bigp, &left_smallp, this](const Photon* p) {
//                update_cube(p->pos, left_bigp, left_smallp);
//            });
//        std::for_each(std::next(photons_pointers.begin(), mid + 1), std::next(photons_pointers.begin(), to),
//            [&right_bigp, &right_smallp, this](const Photon* p) {
//                update_cube(p->pos, right_bigp, right_smallp);
//            });*/
//
//        short left_plane = find_largest_plane(left_bigp, left_smallp);
//        short right_plane = find_largest_plane(right_bigp, right_smallp);
//
//        auto node = new Node(medium, plane);
//        tree.set_at(node_ind, node);
//
//        ftpn_recur.push({ from, mid, left_plane, tree.get_left_ind(node_ind) }); // left
//        ftpn_recur.push({ mid + 1, to, right_plane, tree.get_right_ind(node_ind) }); // right
//    }
//    printf("\tPhotons inserted: " + count);
//    printf("Filling balanced KD-tree (photon map) ended");
//}
//
//__host__ __device__ cpm::PhotonMap::PhotonMap(Type type, Photon* photons, size_t size) : tree(size), depth(0), type(type)
//{
//    //this->settings.ftype = PhotonMapSettings::FilterType::none;
//    //this->type = type;
//    //filters = std::vector<Filter*>(3);
//    //filters[PhotonMapSettings::FilterType::none] = new Filter(1.f);
//    //filters[PhotonMapSettings::FilterType::cone] = new ConeFilter(settings.cf_k);
//    //filters[PhotonMapSettings::FilterType::gaussian] = new GaussianFilter(settings.gf_alpha, settings.gf_beta);
//    //fill_balanced(photons, size);
//}
//
//__host__ __device__ cpm::PhotonMap::~PhotonMap()
//{
//}
//
//