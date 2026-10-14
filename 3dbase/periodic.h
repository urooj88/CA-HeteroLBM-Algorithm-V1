#include "../3dsetup/typedefs.h"

/// Periodic conditions in z direction (front and back)

template<typename VALUE_TYPE, typename INDEX_TYPE>
__global__ void periodicFb(
	VALUE_TYPE *f_post,
	INDEX_TYPE Nnx,
	INDEX_TYPE Nny,
	INDEX_TYPE Nnz){
		
	INDEX_TYPE i = blockIdx.x * blockDim.x + threadIdx.x ;
    INDEX_TYPE j = blockIdx.y * blockDim.y + threadIdx.y ;
	INDEX_TYPE k = blockIdx.z * blockDim.z + threadIdx.z ;
	
	if(i < Nnx && j < Nny && k < Nnz){
		
		INDEX_TYPE ixyz = i + j*Nnx + k*Nnx*Nny;
		INDEX_TYPE ixyz1, ixyz2;
		
		INDEX_TYPE i5 =  ixyz +  5*Nnx*Nny*Nnz;
		INDEX_TYPE i6 =  ixyz +  6*Nnx*Nny*Nnz;
		INDEX_TYPE i9 =  ixyz +  9*Nnx*Nny*Nnz;
		INDEX_TYPE i10 = ixyz + 10*Nnx*Nny*Nnz;
		INDEX_TYPE i11 = ixyz + 11*Nnx*Nny*Nnz;
		INDEX_TYPE i12 = ixyz + 12*Nnx*Nny*Nnz;
		INDEX_TYPE i15 = ixyz + 15*Nnx*Nny*Nnz;
		INDEX_TYPE i16 = ixyz + 16*Nnx*Nny*Nnz;
		INDEX_TYPE i17 = ixyz + 17*Nnx*Nny*Nnz;
		INDEX_TYPE i18 = ixyz + 18*Nnx*Nny*Nnz;
		
		INDEX_TYPE ibck6, ibck10, ibck12, ibck15, ibck17;
		INDEX_TYPE ifnt5, ifnt9,  ifnt11, ifnt16, ifnt18;
		
		// Back wall
		if(k == 0){
			
			ixyz1 = i + j*Nnx + (Nnz-2)*Nnx*Nny;
			
			ifnt5 =  ixyz1 +  5*Nnx*Nny*Nnz;
			ifnt9 =  ixyz1 +  9*Nnx*Nny*Nnz;
			ifnt11 = ixyz1 + 11*Nnx*Nny*Nnz;
			ifnt16 = ixyz1 + 16*Nnx*Nny*Nnz;
			ifnt18 = ixyz1 + 18*Nnx*Nny*Nnz;
			
			f_post[i5] =  f_post[ifnt5];
			f_post[i9] =  f_post[ifnt9];
			f_post[i11] = f_post[ifnt11];
			f_post[i16] = f_post[ifnt16];
			f_post[i18] = f_post[ifnt18];
			
		}
		
		// Front wall
		if(k == Nnz-1){
			
			ixyz2 = i + j*Nnx + 1*Nnx*Nny;
			
			ibck6 =  ixyz2 +  6*Nnx*Nny*Nnz;
			ibck10 = ixyz2 + 10*Nnx*Nny*Nnz;
			ibck12 = ixyz2 + 12*Nnx*Nny*Nnz;
			ibck15 = ixyz2 + 15*Nnx*Nny*Nnz;
			ibck17 = ixyz2 + 17*Nnx*Nny*Nnz;
			
			f_post[i6] =  f_post[ibck6];
			f_post[i10] = f_post[ibck10];
			f_post[i12] = f_post[ibck12];
			f_post[i15] = f_post[ibck15];
			f_post[i17] = f_post[ibck17];
			
		}
	}
}

/// Periodic conditions in x direction (left and right)

template<typename VALUE_TYPE, typename INDEX_TYPE>
__global__ void periodicLr(
	VALUE_TYPE *f_post,
	INDEX_TYPE Nnx,
	INDEX_TYPE Nny,
	INDEX_TYPE Nnz){
		
	INDEX_TYPE i = blockIdx.x * blockDim.x + threadIdx.x ;
    INDEX_TYPE j = blockIdx.y * blockDim.y + threadIdx.y ;
	INDEX_TYPE k = blockIdx.z * blockDim.z + threadIdx.z ;
	
	if(i < Nnx && j < Nny && k < Nnz){
		
		INDEX_TYPE ixyz = i + j*Nnx + k*Nnx*Nny;
		INDEX_TYPE ixyz1, ixyz2;
		
		INDEX_TYPE i1 =  ixyz +  1*Nnx*Nny*Nnz;
		INDEX_TYPE i2 =  ixyz +  2*Nnx*Nny*Nnz;
		INDEX_TYPE i7 =  ixyz +  7*Nnx*Nny*Nnz;
		INDEX_TYPE i8 =  ixyz +  8*Nnx*Nny*Nnz;
		INDEX_TYPE i9 =  ixyz +  9*Nnx*Nny*Nnz;
		INDEX_TYPE i10 = ixyz + 10*Nnx*Nny*Nnz;
		INDEX_TYPE i13 = ixyz + 13*Nnx*Nny*Nnz;
		INDEX_TYPE i14 = ixyz + 14*Nnx*Nny*Nnz;
		INDEX_TYPE i15 = ixyz + 15*Nnx*Nny*Nnz;
		INDEX_TYPE i16 = ixyz + 16*Nnx*Nny*Nnz;
		
		INDEX_TYPE irgt1, irgt7, irgt9,  irgt13, irgt15;
		INDEX_TYPE ilft2, ilft8, ilft10, ilft14, ilft16;
		
		
		// Left wall
		if(i == 0){
			
			ixyz1 = (Nnx-2) + j*Nnx + k*Nnx*Nny;
			
			irgt1 =  ixyz1 +  1*Nnx*Nny*Nnz;
			irgt7 =  ixyz1 +  7*Nnx*Nny*Nnz;
			irgt9 =  ixyz1 +  9*Nnx*Nny*Nnz;
			irgt13 = ixyz1 + 13*Nnx*Nny*Nnz;
			irgt15 = ixyz1 + 15*Nnx*Nny*Nnz;
			
			f_post[i1] =  f_post[irgt1];
			f_post[i7] =  f_post[irgt7];
			f_post[i9] =  f_post[irgt9];
			f_post[i13] = f_post[irgt13];
			f_post[i15] = f_post[irgt15];
			
		}
		
		// Right wall
		if(i == Nnx-1){
			
			ixyz2 = 1 + j*Nnx + k*Nnx*Nny;
			
			ilft2 =  ixyz2 +  2*Nnx*Nny*Nnz;
			ilft8 =  ixyz2 +  8*Nnx*Nny*Nnz;
			ilft10 = ixyz2 + 10*Nnx*Nny*Nnz;
			ilft14 = ixyz2 + 14*Nnx*Nny*Nnz;
			ilft16 = ixyz2 + 16*Nnx*Nny*Nnz;
		
			f_post[i2] =  f_post[ilft2];
			f_post[i8] =  f_post[ilft8];
			f_post[i10] = f_post[ilft10];
			f_post[i14] = f_post[ilft14];
			f_post[i16] = f_post[ilft16];
			
		}
	}
}

/// Periodic conditions in y direction (top and bottom)

template<typename VALUE_TYPE, typename INDEX_TYPE>
__global__ void periodicTb(
	VALUE_TYPE *f_post,
	INDEX_TYPE Nnx,
	INDEX_TYPE Nny,
	INDEX_TYPE Nnz){
		
	INDEX_TYPE i = blockIdx.x * blockDim.x + threadIdx.x ;
    INDEX_TYPE j = blockIdx.y * blockDim.y + threadIdx.y ;
	INDEX_TYPE k = blockIdx.z * blockDim.z + threadIdx.z ;
	
	if(i < Nnx && j < Nny && k < Nnz){
		
		INDEX_TYPE ixyz = i + j*Nnx + k*Nnx*Nny;
		INDEX_TYPE ixyz1, ixyz2;
		
		INDEX_TYPE i3 =  ixyz +  3*Nnx*Nny*Nnz;
		INDEX_TYPE i4 =  ixyz +  4*Nnx*Nny*Nnz;
		INDEX_TYPE i7 =  ixyz +  7*Nnx*Nny*Nnz;
		INDEX_TYPE i8 =  ixyz +  8*Nnx*Nny*Nnz;
		INDEX_TYPE i11 = ixyz + 11*Nnx*Nny*Nnz;
		INDEX_TYPE i12 = ixyz + 12*Nnx*Nny*Nnz;
		INDEX_TYPE i13 = ixyz + 13*Nnx*Nny*Nnz;
		INDEX_TYPE i14 = ixyz + 14*Nnx*Nny*Nnz;
		INDEX_TYPE i17 = ixyz + 17*Nnx*Nny*Nnz;
		INDEX_TYPE i18 = ixyz + 18*Nnx*Nny*Nnz;
		
		INDEX_TYPE ibtm4, ibtm8, ibtm12, ibtm13, ibtm18;
		INDEX_TYPE itop3, itop7, itop11, itop14, itop17;
		
		// Bottom wall
		if(j == 0){
			
			ixyz1 = i + (Nny-2)*Nnx + k*Nnx*Nny;
			
			itop3 =  ixyz1 +  3*Nnx*Nny*Nnz;
			itop7 =  ixyz1 +  7*Nnx*Nny*Nnz;
			itop11 = ixyz1 + 11*Nnx*Nny*Nnz;
			itop14 = ixyz1 + 14*Nnx*Nny*Nnz;
			itop17 = ixyz1 + 17*Nnx*Nny*Nnz;
		
			f_post[i3] =  f_post[itop3];
			f_post[i7] =  f_post[itop7];
			f_post[i11] = f_post[itop11];
			f_post[i14] = f_post[itop14];
			f_post[i17] = f_post[itop17];
			
		}
		
		// Top wall
		if(j == Nny-1){
			
			ixyz2 = i + 1*Nnx + k*Nnx*Nny;
			
			ibtm4 =  ixyz2 +  4*Nnx*Nny*Nnz;
			ibtm8 =  ixyz2 +  8*Nnx*Nny*Nnz;
			ibtm12 = ixyz2 + 12*Nnx*Nny*Nnz;
			ibtm13 = ixyz2 + 13*Nnx*Nny*Nnz;
			ibtm18 = ixyz2 + 18*Nnx*Nny*Nnz;
			
			f_post[i4] =  f_post[ibtm4];
			f_post[i8] =  f_post[ibtm8];
			f_post[i12] = f_post[ibtm12];
			f_post[i13] = f_post[ibtm13];
			f_post[i18] = f_post[ibtm18];
			
		}
	}	
}
