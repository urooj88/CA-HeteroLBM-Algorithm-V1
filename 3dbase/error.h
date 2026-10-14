#include "../3dsetup/typedefs.h"

/// L2 Norm error calculation based on LB velocities

template<typename VALUE_TYPE, typename INDEX_TYPE>
__global__ void error(
	VALUE_TYPE *e1,
	VALUE_TYPE *e2,
	INDEX_TYPE Nnx,
	INDEX_TYPE Nny,
	INDEX_TYPE Nnz,
	VALUE_TYPE *vxerr,
	VALUE_TYPE *vyerr,
	VALUE_TYPE *vzerr,
	VALUE_TYPE *vx,
	VALUE_TYPE *vy,
	VALUE_TYPE *vz){
	
	INDEX_TYPE i = blockIdx.x * blockDim.x + threadIdx.x ;
    INDEX_TYPE j = blockIdx.y * blockDim.y + threadIdx.y ;
	INDEX_TYPE k = blockIdx.z * blockDim.z + threadIdx.z ;
	
	if(i < Nnx && j < Nny && k < Nnz){
	
		INDEX_TYPE ixyz = i + j*Nnx + k*Nnx*Nny;
		
		e1[ixyz] = sqrt(
					(vx[ixyz]-vxerr[ixyz]) * (vx[ixyz]-vxerr[ixyz]) +
					(vy[ixyz]-vyerr[ixyz]) * (vy[ixyz]-vyerr[ixyz]) +
					(vz[ixyz]-vzerr[ixyz]) * (vz[ixyz]-vzerr[ixyz])
					);
					
		e2[ixyz] = sqrt(
					vx[ixyz]*vx[ixyz] + vy[ixyz]*vy[ixyz] + vz[ixyz]*vz[ixyz]
					);
					
		vxerr[ixyz] = vx[ixyz];
		vyerr[ixyz] = vy[ixyz];
		vzerr[ixyz] = vz[ixyz];
		
	}
}
