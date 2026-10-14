#include "../3dbase/constants.h"
#include "../3dsetup/typedefs.h"

/// Multi-relaxation time collision operation for LB nodes

template<typename VALUE_TYPE, typename INDEX_TYPE>
__global__ void collMRT(
	VALUE_TYPE *f,
	VALUE_TYPE *f_post,
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
	
		INDEX_TYPE l;
		VALUE_TYPE m[19];
		
		INDEX_TYPE ixyz = i + j*Nnx + k*Nnx*Nny;
		VALUE_TYPE x = 0;
		
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
		
		m[0] = f[i0] + f[i1] + f[i2] + f[i3] + f[i4] + f[i5] + f[i6] + f[i7] +
			   f[i8] + f[i9] +
			   f[i10] + f[i11] + f[i12] + f[i13] + f[i14] + f[i15] + f[i16] +
			   f[i17] + f[i18];
			 
		m[1] = -30*f[i0] +
			  -11*(f[i1] + f[i2] + f[i3] + f[i4] + f[i5] + f[i6]) +
				8*(f[i7] + f[i8] + f[i9] + f[i10] + f[i11] + f[i12] + f[i13] +
				   f[i14] + f[i15] + f[i16] + f[i17] + f[i18]);
			 
		m[2] = -12*f[i0] +
			   -4*(f[i1] +  f[i2] +  f[i3] +  f[i4] + f[i5] +  f[i6]) +
				   f[i7] +  f[i8] +  f[i9] + f[i10] + f[i11] + f[i12] +
				   f[i13] + f[i14] + f[i15] + f[i16] + f[i17] + f[i18];
			 
		m[3] = 		f[i1] + f[i7] + f[i9] +  f[i11] + f[i13] +
				-1*(f[i2] + f[i8] + f[i10] + f[i12] + f[i14]);
			 
		m[4] = -4*f[i1] + 4*f[i2] + f[i7] + f[i9] + f[i11] + f[i13] +
			   -1*(f[i8] + f[i10] + f[i12] + f[i14]);
			 
		m[5] = 		f[i3] + f[i7] + f[i8] +  f[i15] +  f[i17] +
				-1*(f[i4] + f[i9] + f[i10] + f[i16] + f[i18]);
			 
		m[6] = -4*f[i3] + 4*f[i4] + f[i7] + f[i8] + f[i15] + f[i17] +
			  -1*(f[i9] + f[i10] + f[i16] + f[i18]);
			 
		m[7] = 		f[i5] + f[i11] + f[i12] + f[i15] + f[i16] +
				-1*(f[i6] + f[i13] + f[i14] + f[i17] + f[i18]);
		
		m[8] = -4*f[i5] + 4*f[i6] + f[i11] + f[i12] + f[i15] + f[i16] +
			  -1*(f[i13] + f[i14] + f[i17] + f[i18]);
			 
		m[9] = 2*(f[i1] +  f[i2]) +
			 -1*(f[i3] +  f[i4] + f[i5] +  f[i6]) +
			 f[i7] +  f[i8] +  f[i9] + f[i10] + f[i11] + f[i12] + f[i13] +
			 f[i14] +
			 -2*(f[i15] + f[i16] + f[i17] + f[i18]);
			 
		m[10] = -4*(f[i1] + f[i2]) +
				 2*(f[i3] + f[i4] + f[i5] + f[i6]) +
				 f[i7] + f[i8] + f[i9] + f[i10] + f[i11] + f[i12] + f[i13] +
				 f[i14] +
				 -2*(f[i15] + f[i16] + f[i17] + f[i18]);
			 
		m[11] = 	f[i3] + f[i4] +  f[i7] + f[i8] + f[i9] + f[i10] +
				-1*(f[i5] +  f[i6] + f[i11] + f[i12] + f[i13] + f[i14]);
			 
		m[12] = -2*(f[i3] +  f[i4]) +
				 2*(f[i5] +  f[i6]) +
				 f[i7] +  f[i8] +  f[i9] + f[i10] +
				 -1*(f[i11] + f[i12] + f[i13] + f[i14]);
			 
		m[13] = f[i7] + -1*(f[i8] + f[i9]) + f[i10];
		
		m[14] = f[i15] + -1*(f[i16] + f[i17]) + f[i18];
		
		m[15] = f[i11] + -1*(f[i12] + f[i13]) + f[i14];
		
		m[16] = 	f[i7] + f[i9] + f[i12] + f[i14] +
				-1*(f[i8] + f[i10] + f[i11] + f[i13]);
			
		m[17] = -1*(f[i7] + f[i8] + f[i16] + f[i18]) +
					f[i9] + f[i10] + f[i15] + f[i17];
				
		m[18] = 	f[i11] + f[i12] + f[i17] + f[i18] +
				-1*(f[i13] + f[i14] + f[i15] + f[i16]);
		
		for(l = 0; l < Q; l++){
			if(simtype == 0){
				
				switch(l){
				// The following needs to be updated before using MRT
				case 0: {
					x = rho[ixyz];
					break;}
				case 1: {
					x = rho[ixyz]*
						(-2 + 3*(vx[ixyz]*vx[ixyz]+vy[ixyz]*vy[ixyz]));
					break;}
				case 2: {
					x = rho[ixyz]*(1-3*(vx[ixyz]*vx[ixyz]+vy[ixyz]*vy[ixyz]));
					break;}
				case 3: {
					x = rho[ixyz]*vx[ixyz];
					break;}
				case 4: {
					x = -rho[ixyz]*vx[ixyz];
					break;}
				case 5: {
					x = rho[ixyz]*vy[ixyz];
					break;}
				case 6: {
					x = -rho[ixyz]*vy[ixyz];
					break;}
				case 7: {
					x = rho[ixyz]*(vx[ixyz]*vx[ixyz]-vy[ixyz]*vy[ixyz]);
					break;}
				case 8: {
					x = rho[ixyz]*vx[ixyz]*vy[ixyz];
					break;}
				case 9: {
					x = rho[ixyz]*vx[ixyz]*vy[ixyz];
					break;}
				case 10: {
					x = rho[ixyz]*vx[ixyz]*vy[ixyz];
					break;}
				case 11: {
					x = rho[ixyz]*vx[ixyz]*vy[ixyz];
					break;}
				case 12: {
					x = rho[ixyz]*vx[ixyz]*vy[ixyz];
					break;}
				case 13: {
					x = rho[ixyz]*vx[ixyz]*vy[ixyz];
					break;}
				case 14: {
					x = rho[ixyz]*vx[ixyz]*vy[ixyz];
					break;}
				case 15: {
					x = rho[ixyz]*vx[ixyz]*vy[ixyz];
					break;}
				case 16: {
					x = rho[ixyz]*vx[ixyz]*vy[ixyz];
					break;}
				case 17: {
					x = rho[ixyz]*vx[ixyz]*vy[ixyz];
					break;}
				case 18: {
					x = rho[ixyz]*vx[ixyz]*vy[ixyz];
					break;}
				default: x = 0;}
				
			}else if(simtype == 1){
				switch(l){
				//The following needs to be updated before using MRT
				case 0: {
					x = rho[ixyz];
					break;}
				case 1: {
					x = rho[ixyz]*
						(-2 + 3*(vx[ixyz]*vx[ixyz]+vy[ixyz]*vy[ixyz]));
					break;}
				case 2: {
					x = rho[ixyz]*(1-3*(vx[ixyz]*vx[ixyz]+vy[ixyz]*vy[ixyz]));
					break;}
				case 3: {
					x = rho[ixyz]*vx[ixyz];
					break;}
				case 4: {
					x = -rho[ixyz]*vx[ixyz];
					break;}
				case 5: {
					x = rho[ixyz]*vy[ixyz];
					break;}
				case 6: {
					x = -rho[ixyz]*vy[ixyz];
					break;}
				case 7: {
					x = rho[ixyz]*(vx[ixyz]*vx[ixyz]-vy[ixyz]*vy[ixyz]);
					break;}
				case 8: {
					x = rho[ixyz]*vx[ixyz]*vy[ixyz];
					break;}
				case 9: {
					x = rho[ixyz]*vx[ixyz]*vy[ixyz];
					break;}
				case 10: {
					x = rho[ixyz]*vx[ixyz]*vy[ixyz];
					break;}
				case 11: {
					x = rho[ixyz]*vx[ixyz]*vy[ixyz];
					break;}
				case 12: {
					x = rho[ixyz]*vx[ixyz]*vy[ixyz];
					break;}
				case 13: {
					x = rho[ixyz]*vx[ixyz]*vy[ixyz];
					break;}
				case 14: {
					x = rho[ixyz]*vx[ixyz]*vy[ixyz];
					break;}
				case 15: {
					x = rho[ixyz]*vx[ixyz]*vy[ixyz];
					break;}
				case 16: {
					x = rho[ixyz]*vx[ixyz]*vy[ixyz];
					break;}
				case 17: {
					x = rho[ixyz]*vx[ixyz]*vy[ixyz];
					break;}
				case 18: {
					x = rho[ixyz]*vx[ixyz]*vy[ixyz];
					break;}
				default: x = 0;}
			}
			
			m[k] = m[k] - s[k]*(m[k]-x);
			m[k] /= D[k];
			
		}
		
		f_post[i0] = m[0] + -30*m[1] + 12*m[2];
		
		f_post[i1] = 	m[0] + -11*m[1] + m[3] + 2*m[9] +
					-4*(m[2] + m[4] + m[10]);
		
		f_post[i2] = 	m[0] + -11*m[1] + -4*(m[2] + m[10]) +
					 -1*m[3] + 4*m[4] + 2*m[9];
		
		f_post[i3] = m[0] + -11*m[1] + -4*(m[2] + m[6]) +
					 m[5] + -1*m[9] + 2*m[10] + m[11] + -2*m[12];
		
		f_post[i4] = 	m[0] + -11*m[1] + -4*m[2] + 4*m[6] +
					-1*(m[5] + m[9]) + 2*m[10] + m[11] + -2*m[12] + m[13];
		
		f_post[i5] = 	m[0] + -11*m[1] + -4*(m[2] + m[8]) + m[7] +
					 2*(m[10] + m[12]) + -1*(m[9] + m[11]);
		
		f_post[i6] = 	m[0] + -11*m[1] + -4*m[2] + 4*m[8] +
					2*(m[10] + m[12]) + -1*(m[7] + m[9] + m[11]);
		
		f_post[i7] = m[0] + 8*m[1] + m[2] + m[3] + m[4] + m[5] + m[6] +
					 m[9] + m[10] + m[11] + m[12] + m[13] + m[16] + -1*m[17];
		
		f_post[i8] = m[0] + 8*m[1] + m[2] + m[5] + m[6] + m[9] + m[10] +
					 m[11] + m[12] + -1*(m[3] + m[4] + m[13] + m[16] + m[17]);
		
		f_post[i9] = m[0] + 8*m[1] + m[2] + m[3] + m[4] + m[9] + m[10] +
					 m[11] + m[12] + m[16] + m[17] + -1*(m[5] + m[6] + m[13]);
		
		f_post[i10] = m[0] + 8*m[1] + m[2] + m[9] + m[10] + m[11] + m[12] +
					  m[13] + m[17] + -1*(m[3] + m[4] + m[5] + m[6] + m[16]);
		
		f_post[i11] = m[0] + 8*m[1] + m[2] + m[3] + m[4] + m[7] + m[8] +
					  m[9] + m[10] + m[15] + m[18] + -1*(m[11] + m[12] + m[16]);
		
		f_post[i12] = m[0] + 8*m[1] + m[2] + m[7] + m[8] + m[9] + m[10] +
					  m[16] + m[18] + -1*(m[3] + m[4] + m[11] + m[12] + m[15]);
		
		f_post[i13] = 	m[0] + 8*m[1] + m[2] + m[3] + m[4] + m[9] + m[10] +
					-1*(m[7] + m[8] + m[11] + m[12] + m[15] + m[16] + m[18]);
		
		f_post[i14] = 	m[0] + 8*m[1] + m[2] + m[9] + m[10] + m[15] + m[16] +
					-1*(m[3] + m[4] + m[7] + m[8] + m[11] + m[12] + m[18]);
		
		f_post[i15] = 	m[0] + 8*m[1] + m[2] + m[5] + m[6] + m[7] + m[8] +
					-2*(m[9] + m[10]) + m[14] + m[17] + -1*m[18];
		
		f_post[i16] = 	m[0] + 8*m[1] + m[2] + m[7] + m[8] + -2*(m[9] + m[10]) +
					-1*(m[5] + m[6] + m[14] + m[17] + m[18]);
		
		f_post[i17] = 	m[0] + 8*m[1] + m[2] + m[5] + m[6] + m[17] + m[18] +
					-2*(m[9] + m[10]) + -1*(m[7] + m[8] + m[14]);
		
		f_post[i18] = 	m[0] + 8*m[1] + m[2] + m[14] + m[18] +
					-2*(m[9] + m[10]) + -1*(m[5] + m[6] + m[7] + m[8] + m[17]);
					
	}
}
