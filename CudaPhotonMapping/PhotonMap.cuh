#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "Photon.cuh"
#include "PriorityQueue.cuh"
#include "vec3.cuh"
#include "Tree.cuh"
#include "Pair.cuh"
#include "Stack.cuh"
#include "FourTuple.cuh"
#include "Sort.cuh"
#include <cfloat>
#include "ArrayTools.cuh"
#include "math_functions.cuh"
#include "Photon.cuh"

namespace cpm
{
    class PhotonMap {
    protected:
        struct Node {
            cpm::Photon value;
            short plane;
            __host__ __device__ Node(cpm::Photon* value, short plane) : value(*value), plane(plane) {}
            __host__ __device__ ~Node() {}
        };
        struct NPNode { // TODO как укомплектовать
            const Node* node;
            float sq_dist;
            __host__ __device__ NPNode() : node(nullptr), sq_dist(-1) {}
            __host__ __device__ NPNode(const Node* photon, float sq_dist) : node(photon), sq_dist(sq_dist){}
        };
        struct NPNodeCopmarator {
            __host__ __device__ NPNodeCopmarator() {}
            __host__ __device__ bool operator() (const NPNode* f, const NPNode* s) const {
                return f->sq_dist > s->sq_dist;
            }
        };
        class NPContainerQ : public cpm::priority_queue<NPNode, NPNodeCopmarator> {
            size_t _capacity;
        public:
            __host__ __device__ NPContainerQ(size_t capacity) : _capacity(capacity), cpm::priority_queue<NPNode, NPNodeCopmarator>(capacity) {}
            __host__ __device__ void fpush(const NPNode& elem) {
                priority_queue::push(elem);
            }
            __host__ __device__ void push(const NPNode& elem) {
                printf("trying to push %f", elem.sq_dist);
                // если рассматриваемый больше максимального, то он нам не нужен
                //if (priority_queue::comparator(&priority_queue::top(), &elem)) {
                if(max_dist() < elem.sq_dist){
                    printf("aaa\n");
                    printf("%f\n", priority_queue::top().sq_dist);
                    printf("%f\n", elem.sq_dist);
                    return;
                }
                priority_queue::push(elem); // иначе добавляем
                if (this->size() > _capacity) { // и проверяем, не привысили ли вместимость
                    priority_queue::pop();
                }
            }
            __host__ __device__ void pop() {
                if (priority_queue::size() == 0) {
                    printf("Trying pop from zero size container");
                }
                priority_queue::pop();
            }
            __host__ __device__ size_t capacity() const { return _capacity; }
            __host__ __device__ float max_dist() const { return priority_queue::top().sq_dist; }
            __host__ __device__ float min_dist() const { return 0; }
            __host__ __device__ NPNode& operator[](int index) {
                return priority_queue::heapData[index];
            }
        };
        struct NearestPhotons {
            cpm::vec3 pos;
            cpm::vec3 normal;
            NPContainerQ* container;
            __host__ __device__ NearestPhotons(const cpm::vec3& pos, const cpm::vec3& normal, size_t capacity) : pos(pos), normal(normal) {
                container = new NPContainerQ(capacity);
            }
        };
        struct NPNRNode {
            Node* node;
            int priority;
            size_t heap_ind;
            size_t tree_depth;
            __host__ __device__ NPNRNode() : node(nullptr), priority(0), heap_ind(0), tree_depth(0) {}
            __host__ __device__ NPNRNode(Node* node, size_t heap_ind, size_t tree_depth, int priority)
                : node(node), heap_ind(heap_ind), tree_depth(tree_depth), priority(priority) {}
        };
        struct NPNRNodeComparator {
            bool operator() (const NPNRNode* f, const NPNRNode* s) const {
                return f->priority > s->priority;
            }
        };
        class Filter {
        public:
            /// <summary>
            /// Normalization coefficient
            /// </summary>
            /// <returns></returns>
            const float norm_coef;
            __host__ __device__ Filter(float norm_coef) : norm_coef(norm_coef) {}
            __host__ __device__ float call(const cpm::vec3& ppos, const cpm::vec3& loc, float r) {
                return 1.f;
            }
        };
        class ConeFilter : public Filter {
            float k;
        public:
            __host__ __device__ ConeFilter(float k) : k(k), Filter(1.f - 2 / (3 * k)) {}
            /// <summary>
            /// Возвращает коэфициент для изменения вклада фотона в зависимости от его расстояния до рассматриваемой точки.
            /// </summary>
            /// <param name="ppos">Photon position</param>
            /// <param name="loc">Location where ray hitting a surface</param>
            /// <param name="r">Max distance between loc and nearests photons</param>
            /// <returns></returns>
            __host__ __device__ float call(const cpm::vec3& ppos, const cpm::vec3& loc, float r) {
                float dist = cpm::distance(ppos, loc);
                return 1.f - dist / (k * r);
            }
        };
        class GaussianFilter : public Filter {
            float alpha, beta;
        public:
            __host__ __device__ GaussianFilter(float alpha, float beta) : alpha(alpha), beta(beta), Filter(1.f) {}
            /// <summary>
            /// Возвращает коэфициент для изменения вклада фотона в зависимости от его расстояния до рассматриваемой точки.
            /// </summary>
            /// <param name="ppos">Photon position</param>
            /// <param name="loc">Location where ray hitting a surface</param>
            /// <param name="r">Squared max distance between loc and nearests photons</param>
            /// <returns></returns>
            __host__ __device__ float call(const cpm::vec3& ppos, const cpm::vec3& loc, float r) {
                float dist = cpm::distance(ppos, loc);
                return alpha * (1.f - (1.f - cpm::pow(EXP, -beta * dist * dist / (2 * r))) /
                    (1.f - std::pow(EXP, -beta)));
            }
        };
        struct PhotonComparator {
            int plane;
            __host__ __device__ bool operator()(const cpm::Photon* a, const cpm::Photon* b) const {
                return a->pos[plane] < b->pos[plane];
            }
        };
        cpm::Tree<Node> tree;
        float max_distance = 10000; // TODO перенести в cpp пока что
    public:
        enum Type { def = 0, caustic };

    private:
        Filter* filters[3];
        Type type;
        int depth;
        /// <summary>
        /// Возвращает измерение по которому "куб" имеет наибольшую длину
        /// </summary>
        /// <param name="bigp"> Первая точка, описывающая куб</param>
        /// <param name="smallp">Вторая точка, описывающа куб</param>
        __host__ __device__ short find_largest_plane(const cpm::vec3& bigp, const cpm::vec3& smallp) {
            size_t largest_plane = 0;
            {
                cpm::vec3 dims(
                    fabs(bigp.x - smallp.x),
                    fabs(bigp.y - smallp.y),
                    fabs(bigp.z - smallp.z)
                );
                float max = dims[0];
                for (size_t i = 1; i < 3; i++) {
                    if (dims[i] > max) {
                        max = dims[i];
                        largest_plane = i;
                    }
                }
            }
            return largest_plane;
        }
        __host__ __device__ void update_cube(const cpm::vec3& p, cpm::vec3& bigp, cpm::vec3& smallp) {
            for (size_t point_i = 0; point_i < 3; point_i++) {
                if (p[point_i] > bigp[point_i]) {
                    bigp[point_i] = p[point_i];
                }
                if (p[point_i] < smallp[point_i]) {
                    smallp[point_i] = p[point_i];
                }
            }
        }
        __host__ __device__ void fill_balanced(cpm::Photon* photons, size_t size) {
            if (size == 0) {
                printf("There is nothing to insert\n");
                return;
            }

            printf("Filling balanced KD-tree (photon map) started\n");
            depth = ceil(log2((double)size + 1.0));
            /*
            * нахождение "куба", захватывающего все точки и заполнение вектор указателей для работы с ними
            * + перевод вектор точек в вектор указателей на точки для дальнейшего удобства
            */

            Photon** photons_pointers = (Photon**)(new void* [size]);


            cpm::vec3 bigp(photons[0].pos), smallp(photons[0].pos);
            for (size_t i = 0; i < size; i++) {
                photons_pointers[i] = &photons[i];
                update_cube(photons[i].pos, bigp, smallp);
            }
            /*
            * нахождение измерения с наибольшей длиной
            */
            auto largest_plane = find_largest_plane(bigp, smallp);

            auto photonComp = PhotonComparator();

            size_t count = 0;
            // from, to, plane, node ind
            auto ftpn_recur = cpm::stack<cpm::four_tuple<size_t, size_t, short, int>>(size);
            ftpn_recur.push({ 0, size, largest_plane, 0 });
            while (!ftpn_recur.isEmpty()) {
                short plane;
                size_t from, to;
                int node_ind;
                ftpn_recur.top().tie(&from, &to, &plane, &node_ind);
                ftpn_recur.pop();
                if (to - from == 0) {
                    //node = nullptr;
                    continue;
                }
                count++;

                if (count % ((size / 10)) == 0) {
                    printf("\tPhotons inserted : %d", count);
                }

                if (to - from == 1) {
                    auto node = new Node(photons_pointers[from], plane);
                    tree.set_at(node_ind, node);
                    continue;
                }

                // TODO thurst?
                // Сортировка для нахождения среднего. Сортируем указатели.
                photonComp.plane = plane;
                quick_sort<Photon*, PhotonComparator>(photons_pointers, from, to, photonComp);
                //std::sort(std::next(photons_pointers.begin(), from), std::next(photons_pointers.begin(), to),
                //    [&plane](const Photon* p1, const Photon* p2) { return p1->pos[plane] < p2->pos[plane]; });
                size_t mid = (to - from) / 2 + from;
                Photon* medium = photons_pointers[mid];

                cpm::vec3 left_bigp(-FLT_MAX), left_smallp(FLT_MAX);
                cpm::vec3 right_bigp(left_bigp), right_smallp(left_smallp);
                array_foreach<Photon*>(
                    photons_pointers,
                    from,
                    mid,
                    [&left_bigp, &left_smallp, this](Photon* p) {
                        update_cube(p->pos, left_bigp, left_smallp);
                    }
                );
                array_foreach<Photon*>(
                    photons_pointers,
                    mid + 1,
                    to,
                    [&right_bigp, &right_smallp, this](const Photon* p) {
                        update_cube(p->pos, right_bigp, right_smallp);
                    }
                );


                short left_plane = find_largest_plane(left_bigp, left_smallp);
                short right_plane = find_largest_plane(right_bigp, right_smallp);

                auto node = new Node(medium, plane);
                tree.set_at(node_ind, node);

                ftpn_recur.push({ from, mid, left_plane, tree.get_left_ind(node_ind) }); // left
                ftpn_recur.push({ mid + 1, to, right_plane, tree.get_right_ind(node_ind) }); // right
            }
            printf("\tPhotons inserted: %d\n", count);
            printf("Filling balanced KD-tree (photon map) ended\n");
        }
        __host__ __device__ float calc_dpn(const cpm::vec3& photon_pos, const cpm::vec3& target_photon_pos, const cpm::vec3& target_normal) {
            return cpm::distance(photon_pos, target_photon_pos) * (1.f + 0.f *
                cpm::abs(cpm::vec3::dot(target_normal, cpm::vec3::normalize(photon_pos - target_photon_pos))));
        }
protected:
        __host__ __device__ void locate_q(NearestPhotons& np) {
            int double_depth = 2 * tree.get_depth();

            cpm::stack<cpm::pair<NPNode, size_t>> recur_values(tree.get_depth());

            printf("debug print --------------------------------\n");
            cpm::priority_queue<NPNRNode, NPNRNodeComparator> nodeq(tree.get_depth());
            nodeq.push(NPNRNode(tree.get_root(), 0, 0, 0));
            while (!nodeq.empty()) {
                NPNRNode loc_node = nodeq.top();
                nodeq.pop();
                const Node* p = loc_node.node;
                if (p == nullptr) {
                    continue;
                }
                printf("heap ind %d, ", loc_node.heap_ind);
                float dist1;
                dist1 = np.pos[p->plane] - p->value.pos[p->plane];
                if (tree.is_leaf(loc_node.heap_ind)) {
                    float dpn = calc_dpn(p->value.pos, np.pos, np.normal);
                    np.container->push(NPNode(p, dpn));
                    printf("npc size %d ", np.container->size());
                    printf("np node max dist %f ", np.container->max_dist());
                    printf("leaf!\n");
                    continue;
                }
                if (loc_node.priority > tree.get_depth()) { // TODO or >=
                    while (!recur_values.isEmpty()) {
                        auto pair = recur_values.top();
                        printf("np node max dist %f ", np.container->max_dist());
                        np.container->push(pair.first);
                        printf("npc size %d ", np.container->size());
                        if (tree.is_siblings(loc_node.heap_ind, pair.second)) {
                            break;
                        }
                    }
                }

                size_t left_child_ind = tree.get_left_ind(loc_node.heap_ind);
                size_t right_child_ind = left_child_ind + 1;
                printf("lci %d, ", left_child_ind);
                printf("cap % d, ", tree.get_capacity());
                
                if (dist1 > 0.f) { // if dist1 is positive search right plane
                    if (tree.has_right(loc_node.heap_ind)) {
                        printf("right branch ");
                        nodeq.push(NPNRNode(tree.get_right(loc_node.heap_ind), right_child_ind,
                            loc_node.tree_depth + 1, loc_node.tree_depth + 1));
                    }
                    if (tree.has_left(loc_node.heap_ind) && dist1 * dist1 < np.container->max_dist()) {
                        printf("then right left branch ");
                        nodeq.push(NPNRNode(tree.get_left(loc_node.heap_ind), left_child_ind,
                            loc_node.tree_depth + 1, double_depth - loc_node.tree_depth));
                    }
                }
                else { // dist1 is negative search left first
                    if (tree.has_left(loc_node.heap_ind)) {
                        printf("left branch ");
                        nodeq.push(NPNRNode(tree.get_left(loc_node.heap_ind), left_child_ind,
                            loc_node.tree_depth + 1, loc_node.tree_depth + 1));
                    }
                    if (tree.has_right(loc_node.heap_ind) && dist1 * dist1 < np.container->max_dist()) {
                        printf("then left right branch ");
                        nodeq.push(NPNRNode(tree.get_right(loc_node.heap_ind), right_child_ind,
                            loc_node.tree_depth + 1, double_depth - loc_node.tree_depth));
                    }
                }
                
                // dpn = dp * (1 + f * |cos(nx, x -> p)|) = dp + f * dp * |cos(nx, x->p)|
                float dpn = calc_dpn(p->value.pos, np.pos, np.normal);
                recur_values.push({ NPNode(p, dpn), loc_node.heap_ind });
                printf("put %f ", dpn);
                printf("rc size %d \n", recur_values.get_size());
            }
            printf("prev stack clear debug print \n");
            printf("npc size %d ", recur_values.get_size());
            while (!recur_values.isEmpty()) {
                auto pair = recur_values.top();
                recur_values.pop();
                np.container->push(pair.first);
            }
            printf("final debug print \n");
        }
    public:
        __host__ __device__ PhotonMap(Type type, cpm::Photon* photons, size_t size) : tree(size), depth(0), type(type) {
            fill_balanced(photons, size);
        }
        __host__ __device__ ~PhotonMap() {

        }
    };

}