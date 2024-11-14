#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

namespace cpm
{
	template<typename FirstType, typename SecondType, typename ThirdType, typename FourthType>
	struct four_tuple {
		FirstType first;
		SecondType second;
		ThirdType third;
		FourthType fourth;
		__host__ __device__ four_tuple() {
			first = FirstType();
			second = SecondType();
			third = ThirdType();
			fourth = FourthType();
		}
		__host__ __device__ four_tuple(FirstType first, SecondType second, ThirdType third, FourthType fourth) {
			this->first = first;
			this->second = second;
			this->third = third;
			this->fourth = fourth;
		}
		__host__ __device__ void tie(FirstType* first, SecondType* second, ThirdType* third, FourthType* fourth) {
			*first = this->first;
			*second = this->second;
			*third = this->third;
			*fourth = this->fourth;
		}
	};
}