#pragma once
#include <crt/host_defines.h>
#include "ostream"
#include "iostream"

/* This class can be used by both host and device. Please note that the file is
 * compiled twice, once for the host and once for the device. The latter sets
 * the macro __CUDA_ARCH__ which will make the class use Cuda intrinsics instead
 * of plain computations.
 */

#ifdef __CUDA_ARCH__
#define USE_INTRINSICS
#endif
namespace cpm {
    struct alignas(8) vec2 {
    public:
        __host__ __device__ vec2() {
            _v[0] = 0.f;
            _v[1] = 0.f;
        }

        __host__ __device__ vec2(float v) {
            _v[0] = v;
            _v[1] = v;
        }

        __host__ __device__ vec2(float v1, float v2) {
            _v[0] = v1;
            _v[1] = v2;
        }

        // TODO: Use __saturatef to saturate float values
        __host__ __device__ constexpr bool is_null() const {
            // IEEE-754
            return (
                (_v[0] == 0.f)
                && (_v[1] == 0.f)
                );
        }

        // the following methods just make the code more readable, but they
        // mostly do the same thing
        __host__ __device__ constexpr float u() const { return _v[0]; }
        __host__ __device__ constexpr float v() const { return _v[1]; }
        __host__ __device__ constexpr float x() const { return _v[0]; }
        __host__ __device__ constexpr float y() const { return _v[1]; }
        __host__ __device__ constexpr float r() const { return _v[0]; }
        __host__ __device__ constexpr float g() const { return _v[1]; }

        // +vec2
        __host__ __device__ constexpr const vec2& operator+() const {
            return *this;
        }

        // -vec2 (to negate)
        __host__ __device__ inline vec2 operator-() const {
            return vec2(-_v[0], -_v[1]);
        }

        // vect3[i]
        __host__ __device__ constexpr float operator[](int i) const {
            return _v[i];
        }

        __host__ __device__ constexpr float& operator[](int i) {
            return _v[i];
        };

        __host__ __device__ constexpr vec2& operator+=(const vec2& v) {
#ifdef USE_INTRINSICS
            _v[0] = __fadd_rz(_v[0], v._v[0]);
            _v[1] = __fadd_rz(_v[1], v._v[1]);
#else
            _v[0] += v._v[0];
            _v[1] += v._v[1];
#endif
            return *this;
        }

        __host__ __device__ constexpr vec2& operator-=(const vec2& v) {
#ifdef USE_INTRINSICS
            _v[0] = __fsub_rz(_v[0], v._v[0]);
            _v[1] = __fsub_rz(_v[1], v._v[1]);
#else
            _v[0] -= v._v[0];
            _v[1] -= v._v[1];
#endif
            return *this;
        }

        __host__ __device__ constexpr vec2& operator*=(const vec2& v) {
#ifdef USE_INTRINSICS
            _v[0] = __fmul_rz(_v[0], v._v[0]);
            _v[1] = __fmul_rz(_v[1], v._v[1]);
#else
            _v[0] *= v._v[0];
            _v[1] *= v._v[1];
#endif
            return *this;
        }

        __host__ __device__ constexpr vec2& operator/=(const vec2& v) {
#ifdef USE_INTRINSICS
            _v[0] = __fdiv_rz(_v[0], v._v[0]);
            _v[1] = __fdiv_rz(_v[1], v._v[1]);
#else
            _v[0] /= v._v[0];
            _v[1] /= v._v[1];
#endif
            return *this;
        }

        __host__ __device__ constexpr vec2& operator*=(const float f) {
#ifdef USE_INTRINSICS
            _v[0] = __fmul_rz(_v[0], f);
            _v[1] = __fmul_rz(_v[1], f);
#else
            _v[0] *= f;
            _v[1] *= f;
#endif
            return *this;
        }

        __host__ __device__ constexpr vec2& operator/=(const float f) {
#ifdef USE_INTRINSICS
            float u = __fdiv_rz(1.0f, f);
            _v[0] = __fmul_rz(_v[0], u);
            _v[1] = __fmul_rz(_v[1], u);
#else
            float u = 1.0f / f;
            _v[0] *= u;
            _v[1] *= u;
#endif
            return *this;
        }

        __host__ __device__ float length() const {
#ifdef USE_INTRINSICS
            return
                __fsqrt_rz(
                    __fmul_rz(_v[0], _v[0])
                    + __fmul_rz(_v[1], _v[1])
                );
#else
            return sqrt(
                _v[0] * _v[0] + _v[1] * _v[1]
            );
#endif
        }

        __host__ __device__ constexpr float sq_length() const {
#ifdef USE_INTRINSICS
            return
                (
                    __fmul_rz(_v[0], _v[0])
                    + __fmul_rz(_v[1], _v[1])
                    );
#else
            return (_v[0] * _v[0] + _v[1] * _v[1]);
#endif
        }

        __host__ __device__ inline vec2 gamma_correct() const {
#ifdef USE_INTRINSICS
            return vec2(
                __fsqrt_rz(_v[0]),
                __fsqrt_rz(_v[1])
            );
#else
            return vec2(sqrt(_v[0]), sqrt(_v[1]));
#endif
        }

        __host__ __device__ constexpr void normalize() {
            if (!is_null()) {
                *this /= this->length();
            }
        }

        __host__ __device__ static inline vec2 normalize(vec2 v) {
            if (v.is_null()) {
                return v;
            }
            else {
                return v / v.length();
            }

        }

        __host__ __device__ static inline vec2 abs(vec2 v) {
            if (v.is_null()) {
                return v;
            }

            if (v._v[0] < 0)
                v._v[0] = -v._v[0];
            if (v._v[1] < 0)
                v._v[1] = -v._v[1];
        }

        // dot product
        __host__ __device__ static constexpr float dot(const vec2& v1, const vec2& v2) {
#ifdef USE_INTRINSICS
            return
                (
                    __fmul_rz(v1._v[0], v2._v[0])
                    + __fmul_rz(v1._v[1], v2._v[1])
                    );
#else
            return v1._v[0] * v2._v[0] + v1._v[1] * v2._v[1];
#endif
        }

        __host__ __device__ static inline vec2 cross(const vec2& v1, const vec2& v2) {
#ifdef USE_INTRINSICS
            return
                vec2(
                    __fsub_rz(
                        __fmul_rz(v1._v[1], v2._v[2]),
                        __fmul_rz(v1._v[2], v2._v[1])
                    ),
                    __fmul_rz(
                        __fsub_rz(
                            __fmul_rz(v1._v[0], v2._v[2]),
                            __fmul_rz(v1._v[2], v2._v[0])
                        ),
                        -1
                    )
                );
#else
            return vec2((v1._v[1] * v2._v[2] - v1._v[2] * v2._v[1]),
                (-(v1._v[0] * v2._v[2] - v1._v[2] * v2._v[0])));
#endif
        }

        friend std::ostream& operator<<(std::ostream& os, const vec2& v) {
            os << "(" << v._v[0] << "," << v._v[1] << ")";
            return os;
        }

        friend std::istream& operator>>(std::istream& is, vec2& v) {
            is >> v._v[0] >> v._v[1];
            return is;
        }

        __host__ __device__ friend inline vec2 operator+(const vec2& v1, const vec2& v2) {
#ifdef USE_INTRINSICS
            return
                vec2(
                    __fadd_rz(v1._v[0], v2._v[0]),
                    __fadd_rz(v1._v[1], v2._v[1])
                );
#else
            return vec2(v1._v[0] + v2._v[0], v1._v[1] + v2._v[1]);
#endif
        }

        __host__ __device__ friend inline vec2 operator-(const vec2& v1, const vec2& v2) {
#ifdef USE_INTRINSICS
            return
                vec2(
                    __fsub_rz(v1._v[0], v2._v[0]),
                    __fsub_rz(v1._v[1], v2._v[1])
                );
#else
            return vec2(v1._v[0] - v2._v[0], v1._v[1] - v2._v[1]);
#endif
        }

        __host__ __device__ friend inline vec2 operator*(const vec2& v1, const vec2& v2) {
#ifdef USE_INTRINSICS
            return
                vec2(
                    __fmul_rz(v1._v[0], v2._v[0]),
                    __fmul_rz(v1._v[1], v2._v[1])
                );
#else
            return vec2(v1._v[0] * v2._v[0], v1._v[1] * v2._v[1]);
#endif
        }

        __host__ __device__ friend inline vec2 operator/(const vec2& v1, const vec2& v2) {
#ifdef USE_INTRINSICS
            return
                vec2(
                    __fdiv_rz(v1._v[0], v2._v[0]),
                    __fdiv_rz(v1._v[1], v2._v[1])
                );
#else
            return vec2(v1._v[0] / v2._v[0], v1._v[1] / v2._v[1]);
#endif
        }

        __host__ __device__ friend inline vec2 operator*(float t, const vec2& v) {
#ifdef USE_INTRINSICS
            return
                vec2(
                    __fmul_rz(v._v[0], t),
                    __fmul_rz(v._v[1], t)
                );
#else
            return vec2(t * v._v[0], t * v._v[1]);
#endif
        }

        __host__ __device__ friend inline vec2 operator*(const vec2& v, float t) {
#ifdef USE_INTRINSICS
            return
                vec2(
                    __fmul_rz(v._v[0], t),
                    __fmul_rz(v._v[1], t)
                );
#else
            return vec2(t * v._v[0], t * v._v[1]);
#endif
        }

        __host__ __device__ friend inline vec2 operator/(vec2 v, float t) {
#ifdef USE_INTRINSICS
            return
                vec2(
                    __fdiv_rz(v._v[0], t),
                    __fdiv_rz(v._v[1], t)
                );
#else
            return vec2(v._v[0] / t, v._v[1] / t);
#endif
        }

        __device__ inline vec2 saturate() const {
#ifdef USE_INTRINSICS
            return
                vec2(
                    __saturatef(_v[0]),
                    __saturatef(_v[1])
                );
#else
            return vec2(-1.f, -1.f);
#endif
        }

        __host__ __device__ inline bool operator==(vec2& other) const {
            return x() == other.x() && y() == other.y();
        }
    private:
        float _v[2];
        //float eps = 1e-5;
    };
}