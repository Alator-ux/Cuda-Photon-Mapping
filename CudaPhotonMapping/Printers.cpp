#include "Printers.h"


void println_divider() {
	std::cout << "-----------------" << std::endl;
}

void println_vec3(cpm::vec3 vec3)
{
	std::cout << "x=" << vec3.x << " y=" << vec3.y << " z=" << vec3.z;
}
