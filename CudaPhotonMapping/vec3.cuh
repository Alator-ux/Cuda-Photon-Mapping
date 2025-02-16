#pragma once
#include <crt/host_defines.h>
#include "ostream"
#include "iostream"
#include "vector_types.h"

/* This class can be used by both host and device. Please note that the file is
 * compiled twice, once for the host and once for the device. The latter sets
 * the macro __CUDA_ARCH__ which will make the class use Cuda intrinsics instead
 * of plain computations.
 */

namespace {
    constexpr float eps = 0.001f;
}

#ifdef __CUDA_ARCH__
#define USE_INTRINSICS
#endif
namespace cpm {
    struct alignas(16) vec3 {
        float x, y, z;

        __host__ __device__ vec3() {
            x = 0.f;
            y = 0.f;
            z = 0.f;
        }

        __host__ __device__ vec3(float v) {
            x = v;
            y = v;
            z = v;
        }

        __host__ __device__ vec3(float v1, float v2, float v3) {
            x = v1;
            y = v2;
            z = v3;
        }

        __host__ __device__ vec3(float4 f4) {
            x = f4.x;
            y = f4.y;
            z = f4.z;
        }

        // TODO: Use __saturatef to saturate float values
        __host__ __device__ constexpr bool is_null() const {
            // IEEE-754
            return (
                (x == 0.f)
                && (y == 0.f)
                && (z == 0.f)
                );
        }

        __host__ __device__ constexpr float operator[](int i) const {
            return i == 0 ? x : (i == 1 ? y : z);
        }

        __host__ __device__ constexpr float& operator[](int i) {
            return i == 0 ? x : (i == 1 ? y : z);
        };

        // +vec3
        __host__ __device__ constexpr const vec3& operator+() const {
            return *this;
        }

        // -vec3 (to negate)
        __host__ __device__ inline vec3 operator-() const {
            return vec3(-x, -y, -z);
        }
        
        __host__ __device__ inline void operator=(const vec3& other) {
            x = other.x;
            y = other.y;
            z = other.z;
        }


        __host__ __device__ constexpr vec3& operator+=(const vec3& v) {
#ifdef USE_INTRINSICS
            x = __fadd_rz(x, v.x);
            y = __fadd_rz(y, v.y);
            z = __fadd_rz(z, v.z);
#else
            x += v.x;
            y += v.y;
            z += v.z;
#endif
            return *this;
        }

        __host__ __device__ constexpr vec3& operator-=(const vec3& v) {
#ifdef USE_INTRINSICS
            x = __fsub_rz(x, v.x);
            y = __fsub_rz(y, v.y);
            z = __fsub_rz(z, v.z);
#else
            x -= v.x;
            y -= v.y;
            z -= v.z;
#endif
            return *this;
        }

        __host__ __device__ constexpr vec3& operator*=(const vec3& v) {
#ifdef USE_INTRINSICS
            x = __fmul_rz(x, v.x);
            y = __fmul_rz(y, v.y);
            z = __fmul_rz(z, v.z);
#else
            x *= v.x;
            y *= v.y;
            z *= v.z;
#endif
            return *this;
        }

        __host__ __device__ constexpr vec3& operator/=(const vec3& v) {
#ifdef USE_INTRINSICS
            x = __fdiv_rz(x, v.x);
            y = __fdiv_rz(y, v.y);
            z = __fdiv_rz(z, v.z);
#else
            x /= v.x;
            y /= v.y;
            z /= v.z;
#endif
            return *this;
        }

        __host__ __device__ constexpr vec3& operator*=(const float f) {
#ifdef USE_INTRINSICS
            x = __fmul_rz(x, f);
            y = __fmul_rz(y, f);
            z = __fmul_rz(z, f);
#else
            x *= f;
            y *= f;
            z *= f;
#endif
            return *this;
        }

        __host__ __device__ constexpr vec3& operator/=(const float f) {
#ifdef USE_INTRINSICS
            float u = __fdiv_rz(1.0f, f);
            x = __fmul_rz(x, u);
            y = __fmul_rz(y, u);
            z = __fmul_rz(z, u);
#else
            float u = 1.0f / f;
            x *= u;
            y *= u;
            z *= u;
#endif
            return *this;
        }

        __host__ __device__ float length() const {
#ifdef USE_INTRINSICS
            return
                __fsqrt_rz(
                    __fmul_rz(x, x)
                    + __fmul_rz(y, y)
                    + __fmul_rz(z, z)
                );
#else
            return sqrt(
                x * x + y * y + z * z
            );
#endif
        }

        __host__ __device__ constexpr float sq_length() const {
#ifdef USE_INTRINSICS
            return
                (
                    __fmul_rz(x, x)
                    + __fmul_rz(y, y)
                    + __fmul_rz(z, z)
                    );
#else
            return (x * x + y * y + z * z);
#endif
        }

        __host__ __device__ inline vec3 gamma_correct() const {
#ifdef USE_INTRINSICS
            return vec3(
                __fsqrt_rz(x),
                __fsqrt_rz(y),
                __fsqrt_rz(z)
            );
#else
            return vec3(sqrt(x), sqrt(y), sqrt(z));
#endif
        }

        __host__ __device__ constexpr vec3& normalize() {
            if (!is_null()) {
                *this /= this->length();
            }
            return *this;
        }

        __host__ __device__ static inline vec3 normalize(vec3 v) {
            if (v.is_null()) {
                return v;
            }
            else {
                return v / v.length();
            }
        }

        __host__ __device__ static inline vec3 abs(vec3 v) {
            if (v.is_null()) {
                return v;
            }
            else {
                if (v.x < 0)
                    v.x = -v.x;
                if (v.y < 0)
                    v.y = -v.y;
                if (v.z < 0)
                    v.z = -v.z;
            }
        }

        // dot product
        __host__ __device__ __forceinline__ static constexpr float dot(const vec3& v1, const vec3& v2) {
#ifdef USE_INTRINSICS
            return
                (
                    __fmul_rz(v1.x, v2.x)
                    + __fmul_rz(v1.y, v2.y)
                    + __fmul_rz(v1.z, v2.z)
                    );
#else
            return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
#endif
        }

        __host__ __device__ static inline vec3 cross(const vec3& v1, const vec3& v2) {
#ifdef USE_INTRINSICS
            return
                vec3(
                    __fsub_rz(
                        __fmul_rz(v1.y, v2.z),
                        __fmul_rz(v1.z, v2.y)
                    ),
                    __fmul_rz(
                        __fsub_rz(
                            __fmul_rz(v1.x, v2.z),
                            __fmul_rz(v1.z, v2.x)
                        ),
                        -1
                    ),
                    __fsub_rz(
                        __fmul_rz(v1.x, v2.y),
                        __fmul_rz(v1.y, v2.x)
                    )
                );
#else
            return vec3((v1.y * v2.z - v1.z * v2.y),
                (-(v1.x * v2.z - v1.z * v2.x)),
                (v1.x * v2.y - v1.y * v2.x));
#endif
        }

        friend std::ostream& operator<<(std::ostream& os, const vec3& v) {
            os << "(" << v.x << "," << v.y << "," << v.z << ")";
            return os;
        }

        friend std::istream& operator>>(std::istream& is, vec3& v) {
            is >> v.x >> v.y >> v.z;
            return is;
        }

        __host__ __device__ friend inline vec3 operator+(const vec3& v1, const vec3& v2) {
#ifdef USE_INTRINSICS
            return
                vec3(
                    __fadd_rz(v1.x, v2.x),
                    __fadd_rz(v1.y, v2.y),
                    __fadd_rz(v1.z, v2.z)
                );
#else
            return vec3(v1.x + v2.x, v1.y + v2.y, v1.z + v2.z);
#endif
        }

        __host__ __device__ friend inline vec3 operator-(const vec3& v1, const vec3& v2) {
#ifdef USE_INTRINSICS
            return
                vec3(
                    __fsub_rz(v1.x, v2.x),
                    __fsub_rz(v1.y, v2.y),
                    __fsub_rz(v1.z, v2.z)
                );
#else
            return vec3(v1.x - v2.x, v1.y - v2.y, v1.z - v2.z);
#endif
        }

        __host__ __device__ friend inline vec3 operator*(const vec3& v1, const vec3& v2) {
#ifdef USE_INTRINSICS
            return
                vec3(
                    __fmul_rz(v1.x, v2.x),
                    __fmul_rz(v1.y, v2.y),
                    __fmul_rz(v1.z, v2.z)
                );
#else
            return vec3(v1.x * v2.x, v1.y * v2.y, v1.z * v2.z);
#endif
        }

        __host__ __device__ friend inline vec3 operator/(const vec3& v1, const vec3& v2) {
#ifdef USE_INTRINSICS
            return
                vec3(
                    __fdiv_rz(v1.x, v2.x),
                    __fdiv_rz(v1.y, v2.y),
                    __fdiv_rz(v1.z, v2.z)
                );
#else
            return vec3(v1.x / v2.x, v1.y / v2.y, v1.z / v2.z);
#endif
        }

        __host__ __device__ friend inline vec3 operator*(float t, const vec3& v) {
#ifdef USE_INTRINSICS
            return
                vec3(
                    __fmul_rz(v.x, t),
                    __fmul_rz(v.y, t),
                    __fmul_rz(v.z, t)
                );
#else
            return vec3(t * v.x, t * v.y, t * v.z);
#endif
        }

        __host__ __device__ friend inline vec3 operator*(const vec3& v, float t) {
#ifdef USE_INTRINSICS
            return
                vec3(
                    __fmul_rz(v.x, t),
                    __fmul_rz(v.y, t),
                    __fmul_rz(v.z, t)
                );
#else
            return vec3(t * v.x, t * v.y, t * v.z);
#endif
        }

        __host__ __device__ friend inline vec3 operator/(vec3 v, float t) {
#ifdef USE_INTRINSICS
            return
                vec3(
                    __fdiv_rz(v.x, t),
                    __fdiv_rz(v.y, t),
                    __fdiv_rz(v.z, t)
                );
#else
            return vec3(v.x / t, v.y / t, v.z / t);
#endif
        }

        __device__ inline vec3 saturate() const {
#ifdef USE_INTRINSICS
            return
                vec3(
                    __saturatef(x),
                    __saturatef(y),
                    __saturatef(z)
                );
#else
            return vec3(-1.f, -1.f, -1.f);
#endif
        }
        __host__ __device__ inline vec3 copy() const {
            return vec3(x, y, z);
        }
        __host__ __device__ vec3& add(const vec3& other) {
#ifdef USE_INTRINSICS
            x = __fadd_rz(x, other.x);
            y = __fadd_rz(y, other.y);
            z = __fadd_rz(z, other.z);
#else
            x += other.x;
            y += other.y;
            z += other.z;
#endif
            return *this;
        }
        __host__ __device__ vec3& mult(float t) {
#ifdef USE_INTRINSICS
            x = __fmul_rz(x, t);
            y = __fmul_rz(y, t);
            z = __fmul_rz(z, t);
#else
            x *= t;
            y *= t;
            z *= t;
#endif
            return *this;
        }
        __host__ __device__ vec3& mult(const vec3& other) {
#ifdef USE_INTRINSICS
            x = __fmul_rz(x, other.x);
            y = __fmul_rz(y, other.y);
            z = __fmul_rz(z, other.z);
#else
            x *= other.x;
            y *= other.y;
            z *= other.z;
#endif
            return *this;
        }
        __host__ __device__ vec3& sub(const vec3& other) {
#ifdef USE_INTRINSICS
            x = __fsub_rz(x, other.x);
            y = __fsub_rz(y, other.y);
            z = __fsub_rz(z, other.z);
#else
            x -= other.x;
            y -= other.y;
            z -= other.z;
#endif
            return *this;
        }
        __host__ __device__ vec3& cross(const vec3& other) {
#ifdef USE_INTRINSICS
            x = __fsub_rz(
                __fmul_rz(y, other.z),
                __fmul_rz(z, other.y)
            );
            y = __fmul_rz(
                __fsub_rz(
                    __fmul_rz(x, other.z),
                    __fmul_rz(z, other.x)
                ),
                -1
            );
            z = __fsub_rz(
                __fmul_rz(x, other.y),
                __fmul_rz(y, other.x)
            );
#else
            x = y * other.z - z * other.y;
            y = x * other.z - z * other.x;
            z = x * other.y - y * other.x;
#endif
            return *this;
        }
        __host__ __device__ void clamp_min(float max_value) {
            x = fminf(x, max_value);
            y = fminf(y, max_value);
            z = fminf(z, max_value);
        }
        __host__ __device__ void clamp_max(float min_value) {
            x = fmaxf(x, min_value);
            y = fmaxf(y, min_value);
            z = fmaxf(z, min_value);
        }
        __host__ __device__ inline bool operator==(const vec3& other) const {
            return x == other.x && y == other.y && z && other.z;
        }
        __host__ __device__ inline bool is_zero() const {
            return x == 0.f && y == 0.f && z == 0.f;
        }
        __host__ __device__ inline bool equal(const cpm::vec3& other) {
            bool res = true;
            res = res && fabs(this->x - other.x) < eps;
            res = res && fabs(this->y - other.y) < eps;
            res = res && fabs(this->z - other.z) < eps;
            return res;
        }

        //float eps = 1e-5;
    };
}