#include "Pair.cuh"

__host__ __device__
uint float_as_uint(float f) {
	union {
		float f;
		uint u;
	} u;
	u.f = f;
	return u.u;
}

__host__ __device__
uint64 cpm::to_uint64(cpm::pair<float, float> val) {
	uint a = float_as_uint(val.first);
	uint b = float_as_uint(val.second);
	return (static_cast<uint64>(a) << 32) | b;
}