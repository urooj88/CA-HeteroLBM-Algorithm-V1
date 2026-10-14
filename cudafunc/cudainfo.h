#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <iomanip>
#include <sys/time.h>

#pragma once

/// Gets GPU memory information

void cudaMemInfo(
	size_t &free_m,
	size_t &free_m_update,
	std::string &memallocto,
	size_t &total_m){
	
	cudaMemGetInfo(&free_m_update, &total_m); // Bytes
	
	std::cout<<
	"@@@@@@@@@@\n"<<
	"*CUDAMALLOC - CUDA memory allocated to "<<memallocto<<" (MB): "<<
	(free_m-free_m_update)/(1000*1000)<<
	"\n"<<
	"Free CUDA memory after "<<memallocto<<" (MB): "<<
	free_m_update/(1000*1000)<<
	"\n@@@@@@@@@@"<<std::endl;
	
	free_m = free_m_update;
	
}
