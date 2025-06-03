//#pragma once
//#include <crt/host_defines.h>
//#include <cuda_runtime.h>
//
//#ifndef __CUDA_ARCH__
//
//int atomicAdd(int* ptr, int value) {
//	int old_value = *ptr;
//	*ptr = old_value + value;
//	return old_value;
//}
//
//size_t atomicAdd(size_t* ptr, size_t value) {
//	size_t old_value = *ptr;
//	*ptr = old_value + value;
//	return old_value;
//}
//
//unsigned int atomicAdd(unsigned int* ptr, unsigned int value) {
//	unsigned int old_value = *ptr;
//	*ptr = old_value + value;
//	return old_value;
//}
//
//int atomicSub(int* ptr, int value) {
//	int old_value = *ptr;
//	*ptr = old_value - value;
//	return old_value;
//}
//
//size_t atomicSub(size_t* ptr, size_t value) {
//	size_t old_value = *ptr;
//	*ptr = old_value - value;
//	return old_value;
//}
//
//unsigned int atomicSub(unsigned int* ptr, unsigned int value) {
//	unsigned int old_value = *ptr;
//	*ptr = old_value - value;
//	return old_value;
//}
//
//int atomicExch(int* ptr, int value) {
//	int old_value = *ptr;
//	*ptr = value;
//	return old_value;
//}
//
//size_t atomicExch(size_t* ptr, size_t value) {
//	size_t old_value = *ptr;
//	*ptr = value;
//	return old_value;
//}
//
//unsigned int atomicExch(unsigned int* ptr, unsigned int value) {
//	unsigned int old_value = *ptr;
//	*ptr = value;
//	return old_value;
//}
//
//int atomicCAS(int* ptr, int if_equal_value, int set_value) {
//	int old_value = *ptr;
//	if (old_value == if_equal_value) {
//		*ptr = set_value;
//	}
//	return old_value;
//}
//
//size_t atomicCAS(size_t* ptr, size_t if_equal_value, size_t set_value) {
//	size_t old_value = *ptr;
//	if (old_value == if_equal_value) {
//		*ptr = set_value;
//	}
//	return old_value;
//}
//
//unsigned int atomicCAS(unsigned int* ptr, unsigned int if_equal_value, unsigned int set_value) {
//	unsigned int old_value = *ptr;
//	if (old_value == if_equal_value) {
//		*ptr = set_value;
//	}
//	return old_value;
//}
//
//#endif