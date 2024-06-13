#pragma once
#include <iostream>
template<typename T>
void print_array(T* array, size_t size) {
	for (int i = 0; i < size; i++) {
		std::cout << array[i] << " ";
	}
}

template<typename T>
void println_array(T* array, size_t size) {
	print_array(array, size);
	std::cout << std::endl;
}

void println_divider();