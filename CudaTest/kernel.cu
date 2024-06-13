#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "Test.cuh"

int main(int argc, char * argv[]) {
    setlocale(LC_ALL, "");
    ctest::PQTest();
    ctest::StackTest();
    ctest::PhotonTest();
}