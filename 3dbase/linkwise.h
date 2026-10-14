#include "../3dbase/constants.h"
#include "../3dsetup/typedefs.h"

/// Halfway bounce-back LB boundary conditions for back wall

template<typename VALUE_TYPE, typename INDEX_TYPE>
__global__ void halfwayBouncebackAll(
	VALUE_TYPE *f,
	VALUE_TYPE *f_post,
	VALUE_TYPE latvxback,
	VALUE_TYPE latvyback,
	VALUE_TYPE latvzback,
	VALUE_TYPE latvxbottom,
	VALUE_TYPE latvybottom,
	VALUE_TYPE latvzbottom,
	VALUE_TYPE latvxfront,
	VALUE_TYPE latvyfront,
	VALUE_TYPE latvzfront,
	VALUE_TYPE latvxleft,
	VALUE_TYPE latvyleft,
	VALUE_TYPE latvzleft,
	VALUE_TYPE latvxright,
	VALUE_TYPE latvyright,
	VALUE_TYPE latvzright,
	VALUE_TYPE latvxtop,
	VALUE_TYPE latvytop,
	VALUE_TYPE latvztop,
	INDEX_TYPE Nnx,
	INDEX_TYPE Nny,
	INDEX_TYPE Nnz,
	VALUE_TYPE *rho,
	VALUE_TYPE *vx,
	VALUE_TYPE *vy,
	VALUE_TYPE *vz){
		
	INDEX_TYPE i = blockIdx.x * blockDim.x + threadIdx.x ;
	INDEX_TYPE j = blockIdx.y * blockDim.y + threadIdx.y ;
	INDEX_TYPE k = blockIdx.z * blockDim.z + threadIdx.z ;
	
	if(i < Nnx && j < Nny && k < Nnz){
		
		INDEX_TYPE ixyz = i + j*Nnx + k*Nnx*Nny;
		
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
		
		// Back wall
		if(k == 0 && i != 0 && i != Nnx-1 && j != 0 && j != Nny-1){
			
			f[i5] = f_post[i6] + 6.0 * rho[ixyz] * w[5] *
			(cx[5] * latvxback + cy[5] * latvyback + cz[5] * latvzback);
			
			f[i9] = f_post[i10] + 6.0 * rho[ixyz] * w[9] *
			(cx[9] * latvxback + cy[9] * latvyback + cz[9] * latvzback);
			
			f[i11] = f_post[i12] + 6.0 * rho[ixyz] * w[11] *
			(cx[11] * latvxback + cy[11] * latvyback + cz[11] * latvzback);
			
			f[i16] = f_post[i15] + 6.0 * rho[ixyz] * w[16] *
			(cx[16] * latvxback + cy[16] * latvyback + cz[16] * latvzback);
			
			f[i18] = f_post[i17] + 6.0 * rho[ixyz] * w[18] *
			(cx[18] * latvxback + cy[18] * latvyback + cz[18] * latvzback);
				
		}
		
		// Bottom wall
		if(j == 0 && i != 0 && i != Nnx-1 && k != 0 && j != Nnz-1){
			
			f[i3] = f_post[i4] + 6.0 * rho[ixyz] * w[3] *
			(cx[3] * latvxbottom + cy[3] * latvybottom + cz[3] * latvzbottom);
			
			f[i7] = f_post[i8] + 6.0 * rho[ixyz] * w[7] *
			(cx[7] * latvxbottom + cy[7] * latvybottom + cz[7] * latvzbottom);
			
			f[i11] = f_post[i12] + 6.0 * rho[ixyz] * w[11] *
			(cx[11] * latvxbottom + cy[11] * latvybottom + cz[11] *
			latvzbottom);
			
			f[i14] = f_post[i13] + 6.0 * rho[ixyz] * w[14] *
			(cx[14] * latvxbottom + cy[14] * latvybottom + cz[14] *
			latvzbottom);
			
			f[i17] = f_post[i18] + 6.0 * rho[ixyz] * w[17] *
			(cx[17] * latvxbottom + cy[17] * latvybottom + cz[17] *
			latvzbottom);
				
		}
		
		// Front wall
		if(k == Nnz-1 && i != 0 && i != Nnx-1 && j != 0 && j != Nny-1){
			
			f[i6] = f_post[i5] + 6.0 * rho[ixyz] * w[6] *
			(cx[6] * latvxfront + cy[6] * latvyfront + cz[6] * latvzfront);
			
			f[i10] = f_post[i9] + 6.0 * rho[ixyz] * w[10] *
			(cx[10] * latvxfront + cy[10] * latvyfront + cz[10] * latvzfront);
			
			f[i12] = f_post[i11] + 6.0 * rho[ixyz] * w[12] *
			(cx[12] * latvxfront + cy[12] * latvyfront + cz[12] * latvzfront);
			
			f[i15] = f_post[i16] + 6.0 * rho[ixyz] * w[15] *
			(cx[15] * latvxfront + cy[15] * latvyfront + cz[15] * latvzfront);
			
			f[i17] = f_post[i18] + 6.0 * rho[ixyz] * w[17] *
			(cx[17] * latvxfront + cy[17] * latvyfront + cz[17] * latvzfront);
				
		}
		
		// Left wall
		if(i == 0 && j != 0 && j != Nny-1 && k != 0 && k != Nnz-1){
			
			f[i1] = f_post[i2]+ 6.0 * rho[ixyz] * w[1] *
			(cx[1] * latvxleft + cy[1] * latvyleft + cz[1] * latvzleft);
			
			f[i7] = f_post[i8] + 6.0 * rho[ixyz] * w[7] *
			(cx[7] * latvxleft + cy[7] * latvyleft + cz[7] * latvzleft);
			
			f[i9] = f_post[i10] + 6.0 * rho[ixyz] * w[9] *
			(cx[9] * latvxleft + cy[9] * latvyleft + cz[9] * latvzleft);
			
			f[i13] = f_post[i14] + 6.0 * rho[ixyz] * w[13] *
			(cx[13] * latvxleft + cy[13] * latvyleft + cz[13] * latvzleft);
			
			f[i15] = f_post[i16] + 6.0 * rho[ixyz] * w[15] *
			(cx[15] * latvxleft + cy[15] * latvyleft + cz[15] * latvzleft);
			
		}
		
		// Right wall
		if(i == Nnx-1 && j != 0 && j != Nny-1 && k != 0 && k != Nnz-1){
			
			f[i2] = f_post[i1]+ 6.0 * rho[ixyz] * w[2] *
			(cx[2] * latvxright + cy[2] * latvyright + cz[2] * latvzright);
			
			f[i8] = f_post[i7]+ 6.0 * rho[ixyz] * w[8] *
			(cx[8] * latvxright + cy[8] * latvyright + cz[8] * latvzright);
			
			f[i10] = f_post[i9] + 6.0 * rho[ixyz] * w[10] *
			(cx[10] * latvxright + cy[10] * latvyright + cz[10] * latvzright);
			
			f[i14] = f_post[i13] + 6.0 * rho[ixyz] * w[14] *
			(cx[14] * latvxright + cy[14] * latvyright + cz[14] * latvzright);
			
			f[i16] = f_post[i15] + 6.0 * rho[ixyz] * w[16] *
			(cx[16] * latvxright + cy[16] * latvyright + cz[16] * latvzright);
			
		}
		
		// Top wall
		if(j == Nny-1 && i != 0 && i != Nnx-1 && k != 0 && k != Nnz-1){
			
			f[i4] = f_post[i3] + 6.0 * rho[ixyz] * w[4] *
			(cx[4] * latvxtop + cy[4] * latvytop + cz[4] * latvztop);
			
			f[i8] = f_post[i7] + 6.0 * rho[ixyz] * w[8] *
			(cx[8] * latvxtop + cy[8] * latvytop + cz[8] * latvztop);
			
			f[i12] = f_post[i11] + 6.0 * rho[ixyz] * w[12] *
			(cx[12] * latvxtop + cy[12] * latvytop + cz[12] * latvztop);
			
			f[i13] = f_post[i14] + 6.0 * rho[ixyz] * w[13] * 
			(cx[13] * latvxtop + cy[13] * latvytop + cz[13] * latvztop);
			
			f[i18] = f_post[i17] + 6.0 * rho[ixyz] * w[18] *
			(cx[18] * latvxtop + cy[18] * latvytop + cz[18] * latvztop);
			
		}
		
		// For all the conditions below, velocities used need to be
		// checked! This would be important for moving walls with
		// different velocities.
		
		// Top wall, back left corner
		if(i == 0 && j == Nny-1 && k == 0){
			
			f[i1] = f_post[i2]+ 6.0 * rho[ixyz] * w[1] *
			(cx[1] * latvxtop + cy[1] * latvytop + cz[1] * latvztop);
			
			f[i4] = f_post[i3] + 6.0 * rho[ixyz] * w[4] *
			(cx[4] * latvxtop + cy[4] * latvytop + cz[4] * latvztop);
			
			f[i5] = f_post[i6] + 6.0 * rho[ixyz] * w[5] *
			(cx[5] * latvxtop + cy[5] * latvytop + cz[5] * latvztop);
			
			f[i9] = f_post[i10] + 6.0 * rho[ixyz] * w[9] *
			(cx[9] * latvxtop + cy[9] * latvytop + cz[9] * latvztop);
			
			f[i13] = f_post[i14] + 6.0 * rho[ixyz] * w[13] * 
			(cx[13] * latvxtop + cy[13] * latvytop + cz[13] * latvztop);
			
			f[i18] = f_post[i17] + 6.0 * rho[ixyz] * w[18] *
			(cx[18] * latvxtop + cy[18] * latvytop + cz[18] * latvztop);
			
		}
		
		// Top wall, back edge
		if(i != 0 && i != Nnx-1 && j == Nny-1 && k == 0){
			
			f[i4] = f_post[i3] + 6.0 * rho[ixyz] * w[4] *
			(cx[4] * latvxtop + cy[4] * latvytop + cz[4] * latvztop);
			
			f[i5] = f_post[i6] + 6.0 * rho[ixyz] * w[5] *
			(cx[5] * latvxtop + cy[5] * latvytop + cz[5] * latvztop);
			
			f[i8] = f_post[i7] + 6.0 * rho[ixyz] * w[8] *
			(cx[8] * latvxtop + cy[8] * latvytop + cz[8] * latvztop);
			
			f[i9] = f_post[i10] + 6.0 * rho[ixyz] * w[9] *
			(cx[9] * latvxtop + cy[9] * latvytop + cz[9] * latvztop);
			
			f[i13] = f_post[i14] + 6.0 * rho[ixyz] * w[13] * 
			(cx[13] * latvxtop + cy[13] * latvytop + cz[13] * latvztop);
			
			f[i16] = f_post[i15] + 6.0 * rho[ixyz] * w[16] *
			(cx[16] * latvxtop + cy[16] * latvytop + cz[16] * latvztop);
			
			f[i18] = f_post[i17] + 6.0 * rho[ixyz] * w[18] *
			(cx[18] * latvxtop + cy[18] * latvytop + cz[18] * latvztop);
			
		}
		
		// Top wall, back right corner
		if(i == Nnx-1 && j == Nny-1 && k == 0){
			
			f[i2] = f_post[i1]+ 6.0 * rho[ixyz] * w[2] *
			(cx[2] * latvxtop + cy[2] * latvytop + cz[2] * latvztop);
			
			f[i4] = f_post[i3] + 6.0 * rho[ixyz] * w[4] *
			(cx[4] * latvxtop + cy[4] * latvytop + cz[4] * latvztop);
			
			f[i5] = f_post[i6] + 6.0 * rho[ixyz] * w[5] *
			(cx[5] * latvxtop + cy[5] * latvytop + cz[5] * latvztop);
			
			f[i8] = f_post[i7] + 6.0 * rho[ixyz] * w[8] *
			(cx[8] * latvxtop + cy[8] * latvytop + cz[8] * latvztop);
			
			f[i16] = f_post[i15] + 6.0 * rho[ixyz] * w[16] *
			(cx[16] * latvxtop + cy[16] * latvytop + cz[16] * latvztop);
			
			f[i18] = f_post[i17] + 6.0 * rho[ixyz] * w[18] *
			(cx[18] * latvxtop + cy[18] * latvytop + cz[18] * latvztop);
			
		}
		
		// Top wall, right edge
		if(i == Nnx-1 && j == Nny-1 && k != 0 && k != Nnz-1){
			
			f[i2] = f_post[i1]+ 6.0 * rho[ixyz] * w[2] *
			(cx[2] * latvxtop + cy[2] * latvytop + cz[2] * latvztop);
			
			f[i4] = f_post[i3] + 6.0 * rho[ixyz] * w[4] *
			(cx[4] * latvxtop + cy[4] * latvytop + cz[4] * latvztop);
			
			f[i8] = f_post[i7] + 6.0 * rho[ixyz] * w[8] *
			(cx[8] * latvxtop + cy[8] * latvytop + cz[8] * latvztop);
			
			f[i10] = f_post[i9] + 6.0 * rho[ixyz] * w[10] *
			(cx[10] * latvxtop + cy[10] * latvytop + cz[10] * latvztop);
			
			f[i12] = f_post[i11] + 6.0 * rho[ixyz] * w[12] *
			(cx[12] * latvxtop + cy[12] * latvytop + cz[12] * latvztop);
			
			f[i16] = f_post[i15] + 6.0 * rho[ixyz] * w[16] *
			(cx[16] * latvxtop + cy[16] * latvytop + cz[16] * latvztop);
			
			f[i18] = f_post[i17] + 6.0 * rho[ixyz] * w[18] *
			(cx[18] * latvxtop + cy[18] * latvytop + cz[18] * latvztop);
			
		}
		
		// Top wall, front right corner
		if(i == Nnx-1 && j == Nny-1 && k == Nnz-1){
			
			f[i2] = f_post[i1]+ 6.0 * rho[ixyz] * w[2] *
			(cx[2] * latvxtop + cy[2] * latvytop + cz[2] * latvztop);
			
			f[i4] = f_post[i3] + 6.0 * rho[ixyz] * w[4] *
			(cx[4] * latvxtop + cy[4] * latvytop + cz[4] * latvztop);
			
			f[i6] = f_post[i5] + 6.0 * rho[ixyz] * w[6] *
			(cx[6] * latvxtop + cy[6] * latvytop + cz[6] * latvztop);
			
			f[i8] = f_post[i7] + 6.0 * rho[ixyz] * w[8] *
			(cx[8] * latvxtop + cy[8] * latvytop + cz[8] * latvztop);
			
			f[i10] = f_post[i9] + 6.0 * rho[ixyz] * w[10] *
			(cx[10] * latvxtop + cy[10] * latvytop + cz[10] * latvztop);
			
			f[i12] = f_post[i11] + 6.0 * rho[ixyz] * w[12] *
			(cx[12] * latvxtop + cy[12] * latvytop + cz[12] * latvztop);
			
		}
		
		// Top wall, front edge
		if(i != 0 && i != Nnx-1 && j == Nny-1 && k == Nnz-1){
			
			f[i4] = f_post[i3] + 6.0 * rho[ixyz] * w[4] *
			(cx[4] * latvxtop + cy[4] * latvytop + cz[4] * latvztop);
			
			f[i6] = f_post[i5] + 6.0 * rho[ixyz] * w[6] *
			(cx[6] * latvxtop + cy[6] * latvytop + cz[6] * latvztop);
			
			f[i8] = f_post[i7] + 6.0 * rho[ixyz] * w[8] *
			(cx[8] * latvxtop + cy[8] * latvytop + cz[8] * latvztop);
			
			f[i10] = f_post[i9] + 6.0 * rho[ixyz] * w[10] *
			(cx[10] * latvxtop + cy[10] * latvytop + cz[10] * latvztop);
			
			f[i12] = f_post[i11] + 6.0 * rho[ixyz] * w[12] *
			(cx[12] * latvxtop + cy[12] * latvytop + cz[12] * latvztop);
			
			f[i13] = f_post[i14] + 6.0 * rho[ixyz] * w[13] * 
			(cx[13] * latvxtop + cy[13] * latvytop + cz[13] * latvztop);
			
			f[i15] = f_post[i16] + 6.0 * rho[ixyz] * w[15] *
			(cx[15] * latvxtop + cy[15] * latvytop + cz[15] * latvztop);
			
		}
		
		// Top wall, front left corner
		if(i == 0 && j == Nny-1 && k == Nnz-1){
			
			f[i1] = f_post[i2]+ 6.0 * rho[ixyz] * w[1] *
			(cx[1] * latvxtop + cy[1] * latvytop + cz[1] * latvztop);
			
			f[i4] = f_post[i3] + 6.0 * rho[ixyz] * w[4] *
			(cx[4] * latvxtop + cy[4] * latvytop + cz[4] * latvztop);
			
			f[i6] = f_post[i5] + 6.0 * rho[ixyz] * w[6] *
			(cx[6] * latvxtop + cy[6] * latvytop + cz[6] * latvztop);
			
			f[i12] = f_post[i11] + 6.0 * rho[ixyz] * w[12] *
			(cx[12] * latvxtop + cy[12] * latvytop + cz[12] * latvztop);
			
			f[i13] = f_post[i14] + 6.0 * rho[ixyz] * w[13] * 
			(cx[13] * latvxtop + cy[13] * latvytop + cz[13] * latvztop);
			
			f[i15] = f_post[i16] + 6.0 * rho[ixyz] * w[15] *
			(cx[15] * latvxtop + cy[15] * latvytop + cz[15] * latvztop);
			
		}
		
		// Top wall, left edge
		if(i == 0 && j == Nny-1 && k != 0 && k != Nnz-1){
			
			f[i1] = f_post[i2]+ 6.0 * rho[ixyz] * w[1] *
			(cx[1] * latvxtop + cy[1] * latvytop + cz[1] * latvztop);
			
			f[i4] = f_post[i3] + 6.0 * rho[ixyz] * w[4] *
			(cx[4] * latvxtop + cy[4] * latvytop + cz[4] * latvztop);
			
			f[i9] = f_post[i10] + 6.0 * rho[ixyz] * w[9] *
			(cx[9] * latvxtop + cy[9] * latvytop + cz[9] * latvztop);
			
			f[i12] = f_post[i11] + 6.0 * rho[ixyz] * w[12] *
			(cx[12] * latvxtop + cy[12] * latvytop + cz[12] * latvztop);
			
			f[i13] = f_post[i14] + 6.0 * rho[ixyz] * w[13] * 
			(cx[13] * latvxtop + cy[13] * latvytop + cz[13] * latvztop);
			
			f[i15] = f_post[i16] + 6.0 * rho[ixyz] * w[15] *
			(cx[15] * latvxtop + cy[15] * latvytop + cz[15] * latvztop);
			
			f[i18] = f_post[i17] + 6.0 * rho[ixyz] * w[18] *
			(cx[18] * latvxtop + cy[18] * latvytop + cz[18] * latvztop);
			
		}
		
		// Middle, back left edge
		if(i == 0 && j != 0 && j != Nny-1 && k == 0 ){
			
			f[i1] = f_post[i2]+ 6.0 * rho[ixyz] * w[1] *
			(cx[1] * latvxback + cy[1] * latvyback + cz[1] * latvzback);
			
			f[i5] = f_post[i6] + 6.0 * rho[ixyz] * w[5] *
			(cx[5] * latvxback + cy[5] * latvyback + cz[5] * latvzback);
			
			f[i7] = f_post[i8] + 6.0 * rho[ixyz] * w[7] *
			(cx[7] * latvxback + cy[7] * latvyback + cz[7] * latvzback);
			
			f[i9] = f_post[i10] + 6.0 * rho[ixyz] * w[9] *
			(cx[9] * latvxback + cy[9] * latvyback + cz[9] * latvzback);
			
			f[i11] = f_post[i12] + 6.0 * rho[ixyz] * w[11] *
			(cx[11] * latvxback + cy[11] * latvyback + cz[11] * latvzback);
			
			f[i13] = f_post[i14] + 6.0 * rho[ixyz] * w[13] * 
			(cx[13] * latvxback + cy[13] * latvyback + cz[13] * latvzback);
			
			f[i18] = f_post[i17] + 6.0 * rho[ixyz] * w[18] *
			(cx[18] * latvxback + cy[18] * latvyback + cz[18] * latvzback);
			
		}
		
		// Middle, back right edge
		if(i == Nnx-1 && j != 0 && j != Nny-1 && k == 0 ){
			
			f[i2] = f_post[i1]+ 6.0 * rho[ixyz] * w[2] *
			(cx[2] * latvxback + cy[2] * latvyback + cz[2] * latvzback);
			
			f[i5] = f_post[i6] + 6.0 * rho[ixyz] * w[5] *
			(cx[5] * latvxback + cy[5] * latvyback + cz[5] * latvzback);
			
			f[i8] = f_post[i7]+ 6.0 * rho[ixyz] * w[8] *
			(cx[8] * latvxback + cy[8] * latvyback + cz[8] * latvzback);
			
			f[i11] = f_post[i12] + 6.0 * rho[ixyz] * w[11] *
			(cx[11] * latvxback + cy[11] * latvyback + cz[11] * latvzback);
			
			f[i14] = f_post[i13] + 6.0 * rho[ixyz] * w[14] *
			(cx[14] * latvxback + cy[14] * latvyback + cz[14] * latvzback);
			
			f[i16] = f_post[i15] + 6.0 * rho[ixyz] * w[16] *
			(cx[16] * latvxback + cy[16] * latvyback + cz[16] * latvzback);
			
			f[i18] = f_post[i17] + 6.0 * rho[ixyz] * w[18] *
			(cx[18] * latvxback + cy[18] * latvyback + cz[18] * latvzback);
			
		}
		
		// Middle, front right edge
		if(i == Nnx-1 && j != 0 && j != Nny-1 && k == Nnz-1 ){
			
			f[i2] = f_post[i1]+ 6.0 * rho[ixyz] * w[2] *
			(cx[2] * latvxfront + cy[2] * latvyfront + cz[2] * latvzfront);
			
			f[i6] = f_post[i5] + 6.0 * rho[ixyz] * w[6] *
			(cx[6] * latvxfront + cy[6] * latvyfront + cz[6] * latvzfront);
			
			f[i8] = f_post[i7]+ 6.0 * rho[ixyz] * w[8] *
			(cx[8] * latvxfront + cy[8] * latvyfront + cz[8] * latvzfront);
			
			f[i10] = f_post[i9] + 6.0 * rho[ixyz] * w[10] *
			(cx[10] * latvxfront + cy[10] * latvyfront + cz[10] * latvzfront);
			
			f[i12] = f_post[i11] + 6.0 * rho[ixyz] * w[12] *
			(cx[12] * latvxfront + cy[12] * latvyfront + cz[12] * latvzfront);
			
			f[i14] = f_post[i13] + 6.0 * rho[ixyz] * w[14] *
			(cx[14] * latvxfront + cy[14] * latvyfront + cz[14] * latvzfront);
			
			f[i17] = f_post[i18] + 6.0 * rho[ixyz] * w[17] *
			(cx[17] * latvxfront + cy[17] * latvyfront + cz[17] * latvzfront);
			
		}
		
		// Middle, front left edge
		if(i == 0 && j != 0 && j != Nny-1 && k == Nnz-1 ){
			
			f[i1] = f_post[i2]+ 6.0 * rho[ixyz] * w[1] *
			(cx[1] * latvxfront + cy[1] * latvyfront + cz[1] * latvzfront);
			
			f[i6] = f_post[i5] + 6.0 * rho[ixyz] * w[6] *
			(cx[6] * latvxfront + cy[6] * latvyfront + cz[6] * latvzfront);
			
			f[i7] = f_post[i8] + 6.0 * rho[ixyz] * w[7] *
			(cx[7] * latvxfront + cy[7] * latvyfront + cz[7] * latvzfront);
			
			f[i12] = f_post[i11] + 6.0 * rho[ixyz] * w[12] *
			(cx[12] * latvxfront + cy[12] * latvyfront + cz[12] * latvzfront);
			
			f[i13] = f_post[i14] + 6.0 * rho[ixyz] * w[13] * 
			(cx[13] * latvxfront + cy[13] * latvyfront + cz[13] * latvzfront);
			
			f[i15] = f_post[i16] + 6.0 * rho[ixyz] * w[15] *
			(cx[15] * latvxfront + cy[15] * latvyfront + cz[15] * latvzfront);
			
			f[i17] = f_post[i18] + 6.0 * rho[ixyz] * w[17] *
			(cx[17] * latvxfront + cy[17] * latvyfront + cz[17] * latvzfront);
			
		}
		
		// Bottom wall, back left corner
		if(i == 0 && j == 0 && k == 0){
			
			f[i1] = f_post[i2]+ 6.0 * rho[ixyz] * w[1] *
			(cx[1] * latvxbottom + cy[1] * latvybottom + cz[1] * latvzbottom);
			
			f[i3] = f_post[i4] + 6.0 * rho[ixyz] * w[3] *
			(cx[3] * latvxbottom + cy[3] * latvybottom + cz[3] * latvzbottom);
			
			f[i5] = f_post[i6] + 6.0 * rho[ixyz] * w[5] *
			(cx[5] * latvxbottom + cy[5] * latvybottom + cz[5] * latvzbottom);
			
			f[i7] = f_post[i8] + 6.0 * rho[ixyz] * w[7] *
			(cx[7] * latvxbottom + cy[7] * latvybottom + cz[7] * latvzbottom);
			
			f[i9] = f_post[i10] + 6.0 * rho[ixyz] * w[9] *
			(cx[9] * latvxbottom + cy[9] * latvybottom + cz[9] * latvzbottom);
			
			f[i11] = f_post[i12] + 6.0 * rho[ixyz] * w[11] *
			(cx[11] * latvxbottom + cy[11] * latvybottom + cz[11] *latvzbottom);
			
		}
		
		// Bottom wall, back edge
		if(i != 0 && i != Nnx-1 && j == 0 && k == 0){
			
			f[i3] = f_post[i4] + 6.0 * rho[ixyz] * w[3] *
			(cx[3] * latvxbottom + cy[3] * latvybottom + cz[3] * latvzbottom);
			
			f[i5] = f_post[i6] + 6.0 * rho[ixyz] * w[5] *
			(cx[5] * latvxbottom + cy[5] * latvybottom + cz[5] * latvzbottom);
			
			f[i7] = f_post[i8] + 6.0 * rho[ixyz] * w[7] *
			(cx[7] * latvxbottom + cy[7] * latvybottom + cz[7] * latvzbottom);
			
			f[i9] = f_post[i10] + 6.0 * rho[ixyz] * w[9] *
			(cx[9] * latvxbottom + cy[9] * latvybottom + cz[9] * latvzbottom);
			
			f[i11] = f_post[i12] + 6.0 * rho[ixyz] * w[11] *
			(cx[11] * latvxbottom + cy[11] * latvybottom + cz[11] * latvzbottom);
			
			f[i14] = f_post[i13] + 6.0 * rho[ixyz] * w[14] *
			(cx[14] * latvxbottom + cy[14] * latvybottom + cz[14] * latvzbottom);
			
			f[i16] = f_post[i15] + 6.0 * rho[ixyz] * w[16] *
			(cx[16] * latvxtop + cy[16] * latvytop + cz[16] * latvztop);
			
		}
		
		// Bottom wall, back right corner
		if(i == Nnx-1 && j == 0 && k == 0){
			
			f[i2] = f_post[i1]+ 6.0 * rho[ixyz] * w[2] *
			(cx[2] * latvxbottom + cy[2] * latvybottom + cz[2] * latvzbottom);
			
			f[i3] = f_post[i4] + 6.0 * rho[ixyz] * w[3] *
			(cx[3] * latvxbottom + cy[3] * latvybottom + cz[3] * latvzbottom);
			
			f[i5] = f_post[i6] + 6.0 * rho[ixyz] * w[5] *
			(cx[5] * latvxbottom + cy[5] * latvybottom + cz[5] * latvzbottom);
			
			f[i11] = f_post[i12] + 6.0 * rho[ixyz] * w[11] *
			(cx[11] * latvxbottom + cy[11] * latvybottom + cz[11] * latvzbottom);
			
			f[i14] = f_post[i13] + 6.0 * rho[ixyz] * w[14] *
			(cx[14] * latvxbottom + cy[14] * latvybottom + cz[14] * latvzbottom);
			
			f[i16] = f_post[i15] + 6.0 * rho[ixyz] * w[16] *
			(cx[16] * latvxbottom + cy[16] * latvybottom + cz[16] * latvzbottom);
			
		}
		
		// Bottom wall, right edge
		if(i == Nnx-1 && j == 0 && k != 0 && k != Nnz-1){
			
			f[i2] = f_post[i1]+ 6.0 * rho[ixyz] * w[2] *
			(cx[2] * latvxbottom + cy[2] * latvybottom + cz[2] * latvzbottom);
			
			f[i3] = f_post[i4] + 6.0 * rho[ixyz] * w[3] *
			(cx[3] * latvxbottom + cy[3] * latvybottom + cz[3] * latvzbottom);
			
			f[i10] = f_post[i9] + 6.0 * rho[ixyz] * w[10] *
			(cx[10] * latvxbottom + cy[10] * latvybottom + cz[10] * latvzbottom);
			
			f[i11] = f_post[i12] + 6.0 * rho[ixyz] * w[11] *
			(cx[11] * latvxbottom + cy[11] * latvybottom + cz[11] * latvzbottom);
			
			f[i14] = f_post[i13] + 6.0 * rho[ixyz] * w[14] *
			(cx[14] * latvxbottom + cy[14] * latvybottom + cz[14] * latvzbottom);
			
			f[i16] = f_post[i15] + 6.0 * rho[ixyz] * w[16] *
			(cx[16] * latvxbottom + cy[16] * latvybottom + cz[16] * latvzbottom);
			
			f[i17] = f_post[i18] + 6.0 * rho[ixyz] * w[17] *
			(cx[17] * latvxbottom + cy[17] * latvybottom + cz[17] * latvzbottom);
			
		}
		
		// Bottom wall, front right corner
		if(i == Nnx-1 && j == 0 && k == Nnz-1){
			
			f[i2] = f_post[i1]+ 6.0 * rho[ixyz] * w[2] *
			(cx[2] * latvxbottom + cy[2] * latvybottom + cz[2] * latvzbottom);
			
			f[i3] = f_post[i4] + 6.0 * rho[ixyz] * w[3] *
			(cx[3] * latvxbottom + cy[3] * latvybottom + cz[3] * latvzbottom);
			
			f[i6] = f_post[i5] + 6.0 * rho[ixyz] * w[6] *
			(cx[6] * latvxbottom + cy[6] * latvybottom + cz[6] * latvzbottom);
			
			f[i10] = f_post[i9] + 6.0 * rho[ixyz] * w[10] *
			(cx[10] * latvxbottom + cy[10] * latvybottom + cz[10] * latvzbottom);
			
			f[i14] = f_post[i13] + 6.0 * rho[ixyz] * w[14] *
			(cx[14] * latvxbottom + cy[14] * latvybottom + cz[14] * latvzbottom);
			
			f[i17] = f_post[i18] + 6.0 * rho[ixyz] * w[17] *
			(cx[17] * latvxbottom + cy[17] * latvybottom + cz[17] * latvzbottom);
			
		}
		
		// Bottom wall, front edge
		if(i != 0 && i != Nnx-1 && j == 0 && k == Nnz-1){
			
			f[i3] = f_post[i4] + 6.0 * rho[ixyz] * w[3] *
			(cx[3] * latvxbottom + cy[3] * latvybottom + cz[3] * latvzbottom);
			
			f[i6] = f_post[i5] + 6.0 * rho[ixyz] * w[6] *
			(cx[6] * latvxbottom + cy[6] * latvybottom + cz[6] * latvzbottom);
			
			f[i7] = f_post[i8] + 6.0 * rho[ixyz] * w[7] *
			(cx[7] * latvxbottom + cy[7] * latvybottom + cz[7] * latvzbottom);
			
			f[i10] = f_post[i9] + 6.0 * rho[ixyz] * w[10] *
			(cx[10] * latvxbottom + cy[10] * latvybottom + cz[10] * latvzbottom);
			
			f[i14] = f_post[i13] + 6.0 * rho[ixyz] * w[14] *
			(cx[14] * latvxbottom + cy[14] * latvybottom + cz[14] * latvzbottom);
			
			f[i15] = f_post[i16] + 6.0 * rho[ixyz] * w[15] *
			(cx[15] * latvxbottom + cy[15] * latvybottom + cz[15] * latvzbottom);
			
			f[i17] = f_post[i18] + 6.0 * rho[ixyz] * w[17] *
			(cx[17] * latvxbottom + cy[17] * latvybottom + cz[17] * latvzbottom);
			
		}
		
		// Bottom wall, front left corner
		if(i == 0 && j == 0 && k == Nnz-1){
			
			f[i1] = f_post[i2]+ 6.0 * rho[ixyz] * w[1] *
			(cx[1] * latvxbottom + cy[1] * latvybottom + cz[1] * latvzbottom);
			
			f[i3] = f_post[i4] + 6.0 * rho[ixyz] * w[3] *
			(cx[3] * latvxbottom + cy[3] * latvybottom + cz[3] * latvzbottom);
			
			f[i6] = f_post[i5] + 6.0 * rho[ixyz] * w[6] *
			(cx[6] * latvxbottom + cy[6] * latvybottom + cz[6] * latvzbottom);
			
			f[i7] = f_post[i8] + 6.0 * rho[ixyz] * w[7] *
			(cx[7] * latvxbottom + cy[7] * latvybottom + cz[7] * latvzbottom);
			
			f[i15] = f_post[i16] + 6.0 * rho[ixyz] * w[15] *
			(cx[15] * latvxbottom + cy[15] * latvybottom + cz[15] * latvzbottom);
			
			f[i17] = f_post[i18] + 6.0 * rho[ixyz] * w[17] *
			(cx[17] * latvxbottom + cy[17] * latvybottom + cz[17] * latvzbottom);
			
		}
		
		// Bottom wall, left edge
		if(i == 0 && j == 0 && k != 0 && k != Nnz-1){
			
			f[i1] = f_post[i2]+ 6.0 * rho[ixyz] * w[1] *
			(cx[1] * latvxbottom + cy[1] * latvybottom + cz[1] * latvzbottom);
			
			f[i3] = f_post[i4] + 6.0 * rho[ixyz] * w[3] *
			(cx[3] * latvxbottom + cy[3] * latvybottom + cz[3] * latvzbottom);
			
			f[i7] = f_post[i8] + 6.0 * rho[ixyz] * w[7] *
			(cx[7] * latvxbottom + cy[7] * latvybottom + cz[7] * latvzbottom);
			
			f[i9] = f_post[i10] + 6.0 * rho[ixyz] * w[9] *
			(cx[9] * latvxbottom + cy[9] * latvybottom + cz[9] * latvzbottom);
			
			f[i11] = f_post[i12] + 6.0 * rho[ixyz] * w[11] *
			(cx[11] * latvxbottom + cy[11] * latvybottom + cz[11] * latvzbottom);
			
			f[i15] = f_post[i16] + 6.0 * rho[ixyz] * w[15] *
			(cx[15] * latvxbottom + cy[15] * latvybottom + cz[15] * latvzbottom);
			
			f[i17] = f_post[i18] + 6.0 * rho[ixyz] * w[17] *
			(cx[17] * latvxbottom + cy[17] * latvybottom + cz[17] * latvzbottom);
			
		}
		
	}
}

/// Halfway bounce-back LB boundary conditions for back wall

template<typename VALUE_TYPE, typename INDEX_TYPE>
__global__ void halfwayBouncebackBack(
	VALUE_TYPE *f,
	VALUE_TYPE *f_post,
	VALUE_TYPE latvxback,
	VALUE_TYPE latvyback,
	VALUE_TYPE latvzback,
	INDEX_TYPE Nnx,
	INDEX_TYPE Nny,
	INDEX_TYPE Nnz,
	VALUE_TYPE *rho,
	VALUE_TYPE *vx,
	VALUE_TYPE *vy,
	VALUE_TYPE *vz){
		
	INDEX_TYPE i = blockIdx.x * blockDim.x + threadIdx.x ;
	INDEX_TYPE j = blockIdx.y * blockDim.y + threadIdx.y ;
	INDEX_TYPE k = blockIdx.z * blockDim.z + threadIdx.z ;
	
	if(i < Nnx && j < Nny && k < Nnz){
		
		INDEX_TYPE ixyz = i + j*Nnx + k*Nnx*Nny;
		
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
		
		// Back wall
		if(k == 0){
			
			f[i5] = f_post[i6] + 6.0 * rho[ixyz] * w[5] *
			(cx[5] * latvxback + cy[5] * latvyback + cz[5] * latvzback);
			
			f[i9] = f_post[i10] + 6.0 * rho[ixyz] * w[9] *
			(cx[9] * latvxback + cy[9] * latvyback + cz[9] * latvzback);
			
			f[i11] = f_post[i12] + 6.0 * rho[ixyz] * w[11] *
			(cx[11] * latvxback + cy[11] * latvyback + cz[11] * latvzback);
			
			f[i16] = f_post[i15] + 6.0 * rho[ixyz] * w[16] *
			(cx[16] * latvxback + cy[16] * latvyback + cz[16] * latvzback);
			
			f[i18] = f_post[i17] + 6.0 * rho[ixyz] * w[18] *
			(cx[18] * latvxback + cy[18] * latvyback + cz[18] * latvzback);
				
		}
	}
}

/// Halfway bounce-back LB boundary conditions for bottom wall

template<typename VALUE_TYPE, typename INDEX_TYPE>
__global__ void halfwayBouncebackBottom(
	VALUE_TYPE *f,
	VALUE_TYPE *f_post,
	VALUE_TYPE latvxbottom,
	VALUE_TYPE latvybottom,
	VALUE_TYPE latvzbottom,
	INDEX_TYPE Nnx,
	INDEX_TYPE Nny,
	INDEX_TYPE Nnz,
	VALUE_TYPE *rho,
	VALUE_TYPE *vx,
	VALUE_TYPE *vy,
	VALUE_TYPE *vz){
		
	INDEX_TYPE i = blockIdx.x * blockDim.x + threadIdx.x ;
	INDEX_TYPE j = blockIdx.y * blockDim.y + threadIdx.y ;
	INDEX_TYPE k = blockIdx.z * blockDim.z + threadIdx.z ;
	
	if(i < Nnx && j < Nny && k < Nnz){
		
		INDEX_TYPE ixyz = i + j*Nnx + k*Nnx*Nny;
		
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
		
		// Bottom wall
		if(j == 0){
			
			f[i3] = f_post[i4] + 6.0 * rho[ixyz] * w[3] *
			(cx[3] * latvxbottom + cy[3] * latvybottom + cz[3] * latvzbottom);
			
			f[i7] = f_post[i8] + 6.0 * rho[ixyz] * w[7] *
			(cx[7] * latvxbottom + cy[7] * latvybottom + cz[7] * latvzbottom);
			
			f[i11] = f_post[i12] + 6.0 * rho[ixyz] * w[11] *
			(cx[11] * latvxbottom + cy[11] * latvybottom + cz[11] *
			latvzbottom);
			
			f[i14] = f_post[i13] + 6.0 * rho[ixyz] * w[14] *
			(cx[14] * latvxbottom + cy[14] * latvybottom + cz[14] *
			latvzbottom);
			
			f[i17] = f_post[i18] + 6.0 * rho[ixyz] * w[17] *
			(cx[17] * latvxbottom + cy[17] * latvybottom + cz[17] *
			latvzbottom);
				
		}
	}	
}

/// Halfway bounce-back LB boundary conditions for front wall

template<typename VALUE_TYPE, typename INDEX_TYPE>
__global__ void halfwayBouncebackFront(
	VALUE_TYPE *f,
	VALUE_TYPE *f_post,
	VALUE_TYPE latvxfront,
	VALUE_TYPE latvyfront,
	VALUE_TYPE latvzfront,
	INDEX_TYPE Nnx,
	INDEX_TYPE Nny,
	INDEX_TYPE Nnz,
	VALUE_TYPE *rho,
	VALUE_TYPE *vx,
	VALUE_TYPE *vy,
	VALUE_TYPE *vz){
		
	INDEX_TYPE i = blockIdx.x * blockDim.x + threadIdx.x ;
	INDEX_TYPE j = blockIdx.y * blockDim.y + threadIdx.y ;
	INDEX_TYPE k = blockIdx.z * blockDim.z + threadIdx.z ;
	
	if(i < Nnx && j < Nny && k < Nnz){
		
		INDEX_TYPE ixyz = i + j*Nnx + k*Nnx*Nny;
		
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
		
		// Front wall
		if(k == Nnz-1){
			
			f[i6] = f_post[i5] + 6.0 * rho[ixyz] * w[6] *
			(cx[6] * latvxfront + cy[6] * latvyfront + cz[6] * latvzfront);
			
			f[i10] = f_post[i9] + 6.0 * rho[ixyz] * w[10] *
			(cx[10] * latvxfront + cy[10] * latvyfront + cz[10] * latvzfront);
			
			f[i12] = f_post[i11] + 6.0 * rho[ixyz] * w[12] *
			(cx[12] * latvxfront + cy[12] * latvyfront + cz[12] * latvzfront);
			
			f[i15] = f_post[i16] + 6.0 * rho[ixyz] * w[15] *
			(cx[15] * latvxfront + cy[15] * latvyfront + cz[15] * latvzfront);
			
			f[i17] = f_post[i18] + 6.0 * rho[ixyz] * w[17] *
			(cx[17] * latvxfront + cy[17] * latvyfront + cz[17] * latvzfront);
				
		}
	}	
}

/// Halfway bounce-back LB boundary conditions for left wall

template<typename VALUE_TYPE, typename INDEX_TYPE>
__global__ void halfwayBouncebackLeft(
	VALUE_TYPE *f,
	VALUE_TYPE *f_post,
	VALUE_TYPE latvxleft,
	VALUE_TYPE latvyleft,
	VALUE_TYPE latvzleft,
	INDEX_TYPE Nnx,
	INDEX_TYPE Nny,
	INDEX_TYPE Nnz,
	VALUE_TYPE *rho,
	VALUE_TYPE *vx,
	VALUE_TYPE *vy,
	VALUE_TYPE *vz){
		
	INDEX_TYPE i = blockIdx.x * blockDim.x + threadIdx.x ;
	INDEX_TYPE j = blockIdx.y * blockDim.y + threadIdx.y ;
	INDEX_TYPE k = blockIdx.z * blockDim.z + threadIdx.z ;
	
	if(i < Nnx && j < Nny && k < Nnz){
		
		INDEX_TYPE ixyz = i + j*Nnx + k*Nnx*Nny;
		
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
		
		// Left wall
		if(i == 0){
			
			f[i1] = f_post[i2]+ 6.0 * rho[ixyz] * w[1] *
			(cx[1] * latvxleft + cy[1] * latvyleft + cz[1] * latvzleft);
			
			f[i7] = f_post[i8] + 6.0 * rho[ixyz] * w[7] *
			(cx[7] * latvxleft + cy[7] * latvyleft + cz[7] * latvzleft);
			
			f[i9] = f_post[i10] + 6.0 * rho[ixyz] * w[9] *
			(cx[9] * latvxleft + cy[9] * latvyleft + cz[9] * latvzleft);
			
			f[i13] = f_post[i14] + 6.0 * rho[ixyz] * w[13] *
			(cx[13] * latvxleft + cy[13] * latvyleft + cz[13] * latvzleft);
			
			f[i15] = f_post[i16] + 6.0 * rho[ixyz] * w[15] *
			(cx[15] * latvxleft + cy[15] * latvyleft + cz[15] * latvzleft);
			
		}
	}	
}

/// Halfway bounce-back LB boundary conditions for right wall

template<typename VALUE_TYPE, typename INDEX_TYPE>
__global__ void halfwayBouncebackRight(
	VALUE_TYPE *f,
	VALUE_TYPE *f_post,
	VALUE_TYPE latvxright,
	VALUE_TYPE latvyright,
	VALUE_TYPE latvzright,
	INDEX_TYPE Nnx,
	INDEX_TYPE Nny,
	INDEX_TYPE Nnz,
	VALUE_TYPE *rho,
	VALUE_TYPE *vx,
	VALUE_TYPE *vy,
	VALUE_TYPE *vz){
		
	INDEX_TYPE i = blockIdx.x * blockDim.x + threadIdx.x ;
	INDEX_TYPE j = blockIdx.y * blockDim.y + threadIdx.y ;
	INDEX_TYPE k = blockIdx.z * blockDim.z + threadIdx.z ;
	
	if(i < Nnx && j < Nny && k < Nnz){
		
		INDEX_TYPE ixyz = i + j*Nnx + k*Nnx*Nny;
		
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
		
		// Right wall
		if(i == Nnx-1){
			
			f[i2] = f_post[i1]+ 6.0 * rho[ixyz] * w[2] *
			(cx[2] * latvxright + cy[2] * latvyright + cz[2] * latvzright);
			
			f[i8] = f_post[i7]+ 6.0 * rho[ixyz] * w[8] *
			(cx[8] * latvxright + cy[8] * latvyright + cz[8] * latvzright);
			
			f[i10] = f_post[i9] + 6.0 * rho[ixyz] * w[10] *
			(cx[10] * latvxright + cy[10] * latvyright + cz[10] * latvzright);
			
			f[i14] = f_post[i13] + 6.0 * rho[ixyz] * w[14] *
			(cx[14] * latvxright + cy[14] * latvyright + cz[14] * latvzright);
			
			f[i16] = f_post[i15] + 6.0 * rho[ixyz] * w[16] *
			(cx[16] * latvxright + cy[16] * latvyright + cz[16] * latvzright);
			
		}
	}
}

/// Halfway bounce-back LB boundary conditions for top wall

template<typename VALUE_TYPE, typename INDEX_TYPE>
__global__ void halfwayBouncebackTop(
	VALUE_TYPE *f,
	VALUE_TYPE *f_post,
	VALUE_TYPE latvxtop,
	VALUE_TYPE latvytop,
	VALUE_TYPE latvztop,
	INDEX_TYPE Nnx,
	INDEX_TYPE Nny,
	INDEX_TYPE Nnz,
	VALUE_TYPE *rho,
	VALUE_TYPE *vx,
	VALUE_TYPE *vy,
	VALUE_TYPE *vz){
		
	INDEX_TYPE i = blockIdx.x * blockDim.x + threadIdx.x ;
	INDEX_TYPE j = blockIdx.y * blockDim.y + threadIdx.y ;
	INDEX_TYPE k = blockIdx.z * blockDim.z + threadIdx.z ;
	
	if(i < Nnx && j < Nny && k < Nnz){
		
		INDEX_TYPE ixyz = i + j*Nnx + k*Nnx*Nny;
		
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
		
		// Top wall
		if(j == Nny-1){
			
			f[i4] = f_post[i3] + 6.0 * rho[ixyz] * w[4] *
			(cx[4] * latvxtop + cy[4] * latvytop + cz[4] * latvztop);
			
			f[i8] = f_post[i7] + 6.0 * rho[ixyz] * w[8] *
			(cx[8] * latvxtop + cy[8] * latvytop + cz[8] * latvztop);
			
			f[i12] = f_post[i11] + 6.0 * rho[ixyz] * w[12] *
			(cx[12] * latvxtop + cy[12] * latvytop + cz[12] * latvztop);
			
			f[i13] = f_post[i14] + 6.0 * rho[ixyz] * w[13] * 
			(cx[13] * latvxtop + cy[13] * latvytop + cz[13] * latvztop);
			
			f[i18] = f_post[i17] + 6.0 * rho[ixyz] * w[18] *
			(cx[18] * latvxtop + cy[18] * latvytop + cz[18] * latvztop);
			
		}
	}	
}
