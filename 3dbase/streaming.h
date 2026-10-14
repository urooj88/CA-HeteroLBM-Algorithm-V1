#include "../3dbase/constants.h"
#include "../3dsetup/typedefs.h"

/// Standard streaming for LB nodes

template<typename VALUE_TYPE, typename INDEX_TYPE>
__global__ void streaming(
	VALUE_TYPE *f,
	VALUE_TYPE *f_post,
	INDEX_TYPE Nnx,
	INDEX_TYPE Nny,
	INDEX_TYPE Nnz){
		
	INDEX_TYPE i = blockIdx.x * blockDim.x + threadIdx.x ;
    INDEX_TYPE j = blockIdx.y * blockDim.y + threadIdx.y ;
	INDEX_TYPE k = blockIdx.z * blockDim.z + threadIdx.z ;
	
	if(i < Nnx && j < Nny && k < Nnz){
	
		INDEX_TYPE id, index, index_new, jd, kd, l;
		INDEX_TYPE ixyz = i + j*Nnx + k*Nnx*Nny;
		
		for(l = 0; l < Q; l++){
			
			index = ixyz + l*Nnx*Nny*Nnz;
			
			id = i-cx[l];
			jd = j-cy[l];
			kd = k-cz[l];
			
			index_new = id + jd*Nnx + kd*Nnx*Nny + l*Nnx*Nny*Nnz;
			
			if(id >= 0 && id <= (Nnx-1) &&
			   jd >= 0 && jd <= (Nny-1) &&
			   kd >= 0 && kd <= (Nnz-1)){
				
				f[index] = f_post[index_new];
				
			}
		}
	}
}
