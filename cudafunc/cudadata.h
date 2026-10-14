#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <iomanip>
#include <sys/time.h>

#pragma once

template <typename T>
void inline cuda_Data_Host_to_Device(T** d_,
	const int data_size) {

	(cudaMallocManaged((void**)&(*d_), sizeof(T) * data_size)); CUDA_CHECK_ERROR();
	//cuda_set<<<(data_size+BLOCK_DIM-1)/BLOCK_DIM, BLOCK_DIM>>>( (*d_), 0, data_size   );

	//CUDA_ERROR_CHECK();
	return;
}

template <typename T>
void inline cuda_Data_Host_to_Device(T** d_,
	T* h_,
	const int data_size) {

	(cudaMallocManaged((void**)&(*d_), sizeof(T) * data_size)); CUDA_CHECK_ERROR();
	(cudaMemcpy((*d_), h_, sizeof(T) * data_size, cudaMemcpyHostToDevice)); CUDA_CHECK_ERROR();

	return;
}