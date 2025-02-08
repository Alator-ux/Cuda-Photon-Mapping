#pragma once

#define uint unsigned int

#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )