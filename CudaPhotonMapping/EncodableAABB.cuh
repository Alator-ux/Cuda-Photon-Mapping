#pragma once
#include "AABB.cuh"
#include "Defines.cuh"

class EncodableAABB : public AABB {
	float encode_uint_part(float origin, uint32_t uint_part) {
		uint32_t bits = __builtin_bit_cast(uint32_t, origin);
		uint32_t mask = (1 << (32 - AABB_BITS_TO_ENCODE_IN)) - 1;
		bits &= mask;
		bits |= (uint_part & ((1 << AABB_BITS_TO_ENCODE_IN) - 1));
		return __builtin_bit_cast(float, bits);
	}
	uint32_t decode_uint_part(float origin) {
		uint32_t bits = __builtin_bit_cast(uint32_t, origin);
		return bits & ((1 << AABB_BITS_TO_ENCODE_IN) - 1);
	}
public:
	void encode_uint(uint32_t index) {
		for (int i = 0; i < 6; i++) {
			uint32_t uint_part = (index >> (i * AABB_BITS_TO_ENCODE_IN)) & ((1 << AABB_BITS_TO_ENCODE_IN) - 1);
			corners[i] = encode_uint_part(corners[i], uint_part);
		}
	}
	uint32_t decode_uint() {
		uint32_t value = 0;
		for (int i = 0; i < 6; i++) {
			value |= (decode_uint_part(corners[i]) << (i * 4));
		}
		return value;
	}
};