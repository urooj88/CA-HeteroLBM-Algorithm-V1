#include "../3dbase/constants.h"
#include "../3dsetup/typedefs.h"

/// BGK collision operation for LB nodes

template<typename VALUE_TYPE, typename INDEX_TYPE>
__global__ void collBGK(
	VALUE_TYPE *f,
	VALUE_TYPE *f_post,
	INDEX_TYPE Nnx,
	INDEX_TYPE Nny,
	INDEX_TYPE Nnz,
	VALUE_TYPE *rho,
	VALUE_TYPE rhoinit,
	INDEX_TYPE simtype,
	VALUE_TYPE tau,
	VALUE_TYPE *vx,
	VALUE_TYPE *vy,
	VALUE_TYPE *vz){
	
	INDEX_TYPE i = blockIdx.x * blockDim.x + threadIdx.x ;
    INDEX_TYPE j = blockIdx.y * blockDim.y + threadIdx.y ;
	INDEX_TYPE k = blockIdx.z * blockDim.z + threadIdx.z ;
	
	if(i < Nnx && j < Nny && k < Nnz){
		
		INDEX_TYPE index, ixyz = i + j*Nnx + k*Nnx*Nny, l;
		VALUE_TYPE cu, feq, U2;
		
		for(l = 0; l < Q; l++){
			
			index = ixyz + l*Nnx*Nny*Nnz;
			
			cu = cx[l]*vx[ixyz] + cy[l]*vy[ixyz] + cz[l]*vz[ixyz];
			U2 = vx[ixyz]*vx[ixyz] + vy[ixyz]*vy[ixyz] + vz[ixyz]*vz[ixyz];
			
			if(simtype==0){
				
				feq = w[l] * rho[ixyz] * (1.0 + 3.0*cu + 4.5*cu*cu - 1.5*U2);
				
			}else if(simtype == 1){
				
				feq = w[l] * (rho[ixyz] + 3.0*rhoinit*cu + 4.5*rhoinit*cu*cu -
				1.5*rhoinit*U2);
				
			}
			
			f_post[index] = f[index] - (f[index] - feq)/tau;
			
		}
	}
}
