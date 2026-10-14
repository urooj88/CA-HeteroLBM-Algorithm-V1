#include "../3dsetup/typedefs.h"

/// Calculates LB densities and velocities

template<typename VALUE_TYPE, typename INDEX_TYPE>
__global__ void denVel(
	VALUE_TYPE *f,
	INDEX_TYPE Nnx,
	INDEX_TYPE Nny,
	INDEX_TYPE Nnz,
	VALUE_TYPE *rho,
	VALUE_TYPE rhoinit,
	INDEX_TYPE simtype,
	VALUE_TYPE *vx,
	VALUE_TYPE *vy,
	VALUE_TYPE *vz){
		
	INDEX_TYPE i = blockIdx.x * blockDim.x + threadIdx.x ;
    INDEX_TYPE j = blockIdx.y * blockDim.y + threadIdx.y ;
	INDEX_TYPE k = blockIdx.z * blockDim.z + threadIdx.z ;
	
	if(i < Nnx && j < Nny && k < Nnz){
	
		INDEX_TYPE ixyz = i + j*Nnx + k*Nnx*Nny;
		
		INDEX_TYPE i0 =  ixyz +  0*Nnx*Nny*Nnz;
		INDEX_TYPE i1 =  ixyz +  1*Nnx*Nny*Nnz;
		INDEX_TYPE i2 =  ixyz +  2*Nnx*Nny*Nnz;
		INDEX_TYPE i3 =  ixyz +  3*Nnx*Nny*Nnz;
		INDEX_TYPE i4 =  ixyz +  4*Nnx*Nny*Nnz;
		INDEX_TYPE i5 =  ixyz +  5*Nnx*Nny*Nnz;
		INDEX_TYPE i6 =  ixyz +  6*Nnx*Nny*Nnz;
		INDEX_TYPE i7 =  ixyz +  7*Nnx*Nny*Nnz;
		INDEX_TYPE i8 =  ixyz +  8*Nnx*Nny*Nnz;
		INDEX_TYPE i9 =  ixyz +  9*Nnx*Nny*Nnz;
		INDEX_TYPE i10 = ixyz + 10*Nnx*Nny*Nnz;
		INDEX_TYPE i11 = ixyz + 11*Nnx*Nny*Nnz;
		INDEX_TYPE i12 = ixyz + 12*Nnx*Nny*Nnz;
		INDEX_TYPE i13 = ixyz + 13*Nnx*Nny*Nnz;
		INDEX_TYPE i14 = ixyz + 14*Nnx*Nny*Nnz;
		INDEX_TYPE i15 = ixyz + 15*Nnx*Nny*Nnz;
		INDEX_TYPE i16 = ixyz + 16*Nnx*Nny*Nnz;
		INDEX_TYPE i17 = ixyz + 17*Nnx*Nny*Nnz;
		INDEX_TYPE i18 = ixyz + 18*Nnx*Nny*Nnz;
		
		rho[ixyz] = f[i0]  + f[i1]  + f[i2] +  f[i3] +  f[i4] +  f[i5] +
					f[i6]  + f[i7]  + f[i8] +  f[i9] +  f[i10] + f[i11] +
					f[i12] + f[i13] + f[i14] + f[i15] + f[i16] + f[i17] +
					f[i18];
					
		if(simtype == 0){
			
			vx[ixyz] = (f[i1]+f[i7]+f[i9]+f[i13]+f[i15]-
						f[i2]-f[i8]-f[i10]-f[i14]-f[i16])/rho[ixyz];
			
			vy[ixyz] = (f[i3]+f[i7]+f[i11]+f[i14]+f[i17]-
						f[i4]-f[i8]-f[i12]-f[i13]-f[i18])/rho[ixyz];
			
			vz[ixyz] = (f[i5]+f[i9]+f[i11]+f[i16]+f[i18]-
						f[i6]-f[i10]-f[i12]-f[i15]-f[i17])/rho[ixyz];
			
		}else if(simtype == 1){
			
			vx[ixyz] = (f[i1]+f[i7]+f[i9]+f[i13]+f[i15]-
						f[i2]-f[i8]-f[i10]-f[i14]-f[i16])/rhoinit;
			
			vy[ixyz] = (f[i3]+f[i7]+f[i11]+f[i14]+f[i17]-
						f[i4]-f[i8]-f[i12]-f[i13]-f[i18])/rhoinit;
			
			vz[ixyz] = (f[i5]+f[i9]+f[i11]+f[i16]+f[i18]-
						f[i6]-f[i10]-f[i12]-f[i15]-f[i17])/rhoinit;
			
		}
		
		/*if(std::isnan(vx[ixyz]) || std::isnan(vy[ixyz]) || std::isnan(vz[ixyz])){
			printf("\nVelocity NaNs at %d, check system!\n",ixyz);			
		}*/
		
	}
}
