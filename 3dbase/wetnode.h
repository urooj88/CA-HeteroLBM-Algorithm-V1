#include "../3dbase/constants.h"
#include "../3dsetup/typedefs.h"

/// Non-equilibrium extrapolation boundary condition for LB nodes on left wall
/// - specifies velocities and uses zero-gradient density

template<typename VALUE_TYPE, typename INDEX_TYPE>
__global__ void nonEqExLeft(
	VALUE_TYPE *f,
	VALUE_TYPE latvxleft,
	VALUE_TYPE latvyleft,
	VALUE_TYPE latvzleft,
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
	
		INDEX_TYPE kd, jd, id;
		INDEX_TYPE ixyzd;
		VALUE_TYPE rhofeq, vxfeq, vyfeq, vzfeq;
		INDEX_TYPE l;
		VALUE_TYPE cu, U2;
		VALUE_TYPE feq;
		VALUE_TYPE FMf, FMb;
		INDEX_TYPE index, indexidjdkd;
		
		INDEX_TYPE ixyz = i + j*Nnx + k*Nnx*Nny;
		
		// Left wall
		if(i == 0){
			
			kd = k; jd = j; id = 1;
			ixyzd = id + jd*Nnx + kd*Nnx*Nny;
			
			l = 1; //********************
			
			rhofeq = rho[ixyzd];
			vxfeq = latvxleft; vyfeq = latvyleft; vzfeq = latvzleft;
			
			cu = cx[l]*vxfeq + cy[l]*vyfeq + cz[l]*vzfeq;
			U2 = vxfeq*vxfeq + vyfeq*vyfeq + vzfeq*vzfeq;
			
			if(simtype == 0){
				
				feq = w[l] * rhofeq * (1.0 + 3.0*cu + 4.5*cu*cu - 1.5*U2);
				
			}else if(simtype == 1){
				
				feq = w[l] * (rhofeq + 3.0*rhoinit*cu + 4.5*rhoinit*cu*cu -
				1.5*rhoinit*U2);
				
			}
			
			FMb = feq;
			
			rhofeq = rho[ixyzd]; vxfeq = vx[ixyzd]; vyfeq = vy[ixyzd];
			vzfeq = vz[ixyzd];
			
			cu = cx[l]*vxfeq + cy[l]*vyfeq + cz[l]*vzfeq;
			U2 = vxfeq*vxfeq + vyfeq*vyfeq + vzfeq*vzfeq;
			
			if(simtype == 0){
				
				feq = w[l] * rhofeq * (1.0 + 3.0*cu + 4.5*cu*cu - 1.5*U2);
				
			}else if(simtype == 1){
				
				feq = w[l] * (rhofeq + 3.0*rhoinit*cu + 4.5*rhoinit*cu*cu -
				1.5*rhoinit*U2);
				
			}
			
			FMf = feq;
			
			index = ixyz + l*Nnx*Nny*Nnz;
			indexidjdkd = ixyzd + l*Nnx*Nny*Nnz;
			
			f[index]=FMb+(f[indexidjdkd]-FMf);
			
			l = 7; //********************
			
			rhofeq = rho[ixyzd];
			vxfeq = latvxleft; vyfeq = latvyleft; vzfeq = latvzleft;
			
			cu = cx[l]*vxfeq + cy[l]*vyfeq + cz[l]*vzfeq;
			U2 = vxfeq*vxfeq + vyfeq*vyfeq + vzfeq*vzfeq;
			
			if(simtype == 0){
				
				feq = w[l] * rhofeq * (1.0 + 3.0*cu + 4.5*cu*cu - 1.5*U2);
				
			}else if(simtype == 1){
				
				feq = w[l] * (rhofeq + 3.0*rhoinit*cu + 4.5*rhoinit*cu*cu -
				1.5*rhoinit*U2);
				
			}
			
			FMb = feq;
			
			rhofeq = rho[ixyzd]; vxfeq = vx[ixyzd]; vyfeq = vy[ixyzd];
			vzfeq = vz[ixyzd];
			
			cu = cx[l]*vxfeq + cy[l]*vyfeq + cz[l]*vzfeq;
			U2 = vxfeq*vxfeq + vyfeq*vyfeq + vzfeq*vzfeq;
			
			if(simtype == 0){
				
				feq = w[l] * rhofeq * (1.0 + 3.0*cu + 4.5*cu*cu - 1.5*U2);
				
			}else if(simtype == 1){
				
				feq = w[l] * (rhofeq + 3.0*rhoinit*cu + 4.5*rhoinit*cu*cu -
				1.5*rhoinit*U2);
				
			}
			
			FMf = feq;
			
			index = ixyz + l*Nnx*Nny*Nnz;
			indexidjdkd = ixyzd + l*Nnx*Nny*Nnz;
			
			f[index]=FMb+(f[indexidjdkd]-FMf);
			
			l = 9; //********************
			
			rhofeq = rho[ixyzd];
			vxfeq = latvxleft; vyfeq = latvyleft; vzfeq = latvzleft;
			
			cu = cx[l]*vxfeq + cy[l]*vyfeq + cz[l]*vzfeq;
			U2 = vxfeq*vxfeq + vyfeq*vyfeq + vzfeq*vzfeq;
			
			if(simtype == 0){
				
				feq = w[l] * rhofeq * (1.0 + 3.0*cu + 4.5*cu*cu - 1.5*U2);
				
			}else if(simtype == 1){
				
				feq = w[l] * (rhofeq + 3.0*rhoinit*cu + 4.5*rhoinit*cu*cu -
				1.5*rhoinit*U2);
				
			}
			
			FMb = feq;
			
			rhofeq = rho[ixyzd]; vxfeq = vx[ixyzd]; vyfeq = vy[ixyzd];
			vzfeq = vz[ixyzd];
			
			cu = cx[l]*vxfeq + cy[l]*vyfeq + cz[l]*vzfeq;
			U2 = vxfeq*vxfeq + vyfeq*vyfeq + vzfeq*vzfeq;
			
			if(simtype == 0){
				
				feq = w[l] * rhofeq * (1.0 + 3.0*cu + 4.5*cu*cu - 1.5*U2);
				
			}else if(simtype == 1){
				
				feq = w[l] * (rhofeq + 3.0*rhoinit*cu + 4.5*rhoinit*cu*cu -
				1.5*rhoinit*U2);
				
			}
			
			FMf = feq;
			
			index = ixyz + l*Nnx*Nny*Nnz;
			indexidjdkd = ixyzd + l*Nnx*Nny*Nnz;
			
			f[index]=FMb+(f[indexidjdkd]-FMf);
			
			l = 13; //********************
			
			rhofeq = rho[ixyzd];
			vxfeq = latvxleft; vyfeq = latvyleft; vzfeq = latvzleft;
			
			cu = cx[l]*vxfeq + cy[l]*vyfeq + cz[l]*vzfeq;
			U2 = vxfeq*vxfeq + vyfeq*vyfeq + vzfeq*vzfeq;
			
			if(simtype == 0){
				
				feq = w[l] * rhofeq * (1.0 + 3.0*cu + 4.5*cu*cu - 1.5*U2);
				
			}else if(simtype == 1){
				
				feq = w[l] * (rhofeq + 3.0*rhoinit*cu + 4.5*rhoinit*cu*cu -
				1.5*rhoinit*U2);
				
			}
			
			FMb = feq;
			
			rhofeq = rho[ixyzd]; vxfeq = vx[ixyzd]; vyfeq = vy[ixyzd];
			vzfeq = vz[ixyzd];
			
			cu = cx[l]*vxfeq + cy[l]*vyfeq + cz[l]*vzfeq;
			U2 = vxfeq*vxfeq + vyfeq*vyfeq + vzfeq*vzfeq;
			
			if(simtype == 0){
				
				feq = w[l] * rhofeq * (1.0 + 3.0*cu + 4.5*cu*cu - 1.5*U2);
				
			}else if(simtype == 1){
				
				feq = w[l] * (rhofeq + 3.0*rhoinit*cu + 4.5*rhoinit*cu*cu -
				1.5*rhoinit*U2);
				
			}
			
			FMf = feq;
			
			index = ixyz + l*Nnx*Nny*Nnz;
			indexidjdkd = ixyzd + l*Nnx*Nny*Nnz;
			
			f[index]=FMb+(f[indexidjdkd]-FMf);
			
			l = 15; //********************
			
			rhofeq = rho[ixyzd];
			vxfeq = latvxleft; vyfeq = latvyleft; vzfeq = latvzleft;
			
			cu = cx[l]*vxfeq + cy[l]*vyfeq + cz[l]*vzfeq;
			U2 = vxfeq*vxfeq + vyfeq*vyfeq + vzfeq*vzfeq;
			
			if(simtype == 0){
				
				feq = w[l] * rhofeq * (1.0 + 3.0*cu + 4.5*cu*cu - 1.5*U2);
				
			}else if(simtype == 1){
				
				feq = w[l] * (rhofeq + 3.0*rhoinit*cu + 4.5*rhoinit*cu*cu -
				1.5*rhoinit*U2);
				
			}
			
			FMb = feq;
			
			rhofeq = rho[ixyzd]; vxfeq = vx[ixyzd]; vyfeq = vy[ixyzd];
			vzfeq = vz[ixyzd];
			
			cu = cx[l]*vxfeq + cy[l]*vyfeq + cz[l]*vzfeq;
			U2 = vxfeq*vxfeq + vyfeq*vyfeq + vzfeq*vzfeq;
			
			if(simtype == 0){
				
				feq = w[l] * rhofeq * (1.0 + 3.0*cu + 4.5*cu*cu - 1.5*U2);
				
			}else if(simtype == 1){
				
				feq = w[l] * (rhofeq + 3.0*rhoinit*cu + 4.5*rhoinit*cu*cu -
				1.5*rhoinit*U2);
				
			}
			
			FMf = feq;
			
			index = ixyz + l*Nnx*Nny*Nnz;
			indexidjdkd = ixyzd + l*Nnx*Nny*Nnz;
			
			f[index]=FMb+(f[indexidjdkd]-FMf);
			
		}
	}
}

/// Non-equilibrium extrapolation boundary condition for LB nodes on right wall
/// - uses density of 1 and zero-gradient velocities

template<typename VALUE_TYPE, typename INDEX_TYPE>
__global__ void nonEqExRight(
	VALUE_TYPE *f,
	VALUE_TYPE latvxright,
	VALUE_TYPE latvyright,
	VALUE_TYPE latvzright,
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
	
		INDEX_TYPE kd, jd, id;
		INDEX_TYPE ixyzd;
		VALUE_TYPE rhofeq, vxfeq, vyfeq, vzfeq;
		INDEX_TYPE l;
		VALUE_TYPE cu, U2;
		VALUE_TYPE feq;
		VALUE_TYPE FMf, FMb;
		INDEX_TYPE index, indexidjdkd;
		
		INDEX_TYPE ixyz = i + j*Nnx + k*Nnx*Nny;
		
		// Right wall
		if(i = Nnx-1){
			
			kd = k; jd = j; id = Nnx-2;
			ixyzd = id + jd*Nnx + kd*Nnx*Nny;
			
			l = 2; //********************
			
			rhofeq = 1.0;
			vxfeq = vx[ixyzd]; vyfeq = vy[ixyzd]; vzfeq = vz[ixyzd];
			
			cu = cx[l]*vxfeq + cy[l]*vyfeq + cz[l]*vzfeq;
			U2 = vxfeq*vxfeq + vyfeq*vyfeq + vzfeq*vzfeq;
			
			if(simtype == 0){
				
				feq = w[l] * rhofeq * (1.0 + 3.0*cu + 4.5*cu*cu - 1.5*U2);
				
			}else if(simtype == 1){
				
				feq = w[l] * (rhofeq + 3.0*rhoinit*cu + 4.5*rhoinit*cu*cu -
				1.5*rhoinit*U2);
				
			}
			
			FMb = feq;
			
			rhofeq = rho[ixyzd]; vxfeq = vx[ixyzd]; vyfeq = vy[ixyzd];
			vzfeq = vz[ixyzd];
			
			cu = cx[l]*vxfeq + cy[l]*vyfeq + cz[l]*vzfeq;
			U2 = vxfeq*vxfeq + vyfeq*vyfeq + vzfeq*vzfeq;
			
			if(simtype == 0){
				
				feq = w[l] * rhofeq * (1.0 + 3.0*cu + 4.5*cu*cu - 1.5*U2);
				
			}else if(simtype == 1){
				
				feq = w[l] * (rhofeq + 3.0*rhoinit*cu + 4.5*rhoinit*cu*cu -
				1.5*rhoinit*U2);
				
			}
			
			FMf = feq;
			
			index = ixyz + l*Nnx*Nny*Nnz;
			indexidjdkd = ixyzd + l*Nnx*Nny*Nnz;
			
			f[index]=FMb+(f[indexidjdkd]-FMf);
			
			l = 8; //********************
			
			rhofeq = 1.0;
			vxfeq = vx[ixyzd]; vyfeq = vy[ixyzd]; vzfeq = vz[ixyzd];
			
			cu = cx[l]*vxfeq + cy[l]*vyfeq + cz[l]*vzfeq;
			U2 = vxfeq*vxfeq + vyfeq*vyfeq + vzfeq*vzfeq;
			
			if(simtype == 0){
				
				feq = w[l] * rhofeq * (1.0 + 3.0*cu + 4.5*cu*cu - 1.5*U2);
				
			}else if(simtype == 1){
				
				feq = w[l] * (rhofeq + 3.0*rhoinit*cu + 4.5*rhoinit*cu*cu -
				1.5*rhoinit*U2);
				
			}
			
			FMb = feq;
			
			rhofeq = rho[ixyzd]; vxfeq = vx[ixyzd]; vyfeq = vy[ixyzd];
			vzfeq = vz[ixyzd];
			
			cu = cx[l]*vxfeq + cy[l]*vyfeq + cz[l]*vzfeq;
			U2 = vxfeq*vxfeq + vyfeq*vyfeq + vzfeq*vzfeq;
			
			if(simtype == 0){
				
				feq = w[l] * rhofeq * (1.0 + 3.0*cu + 4.5*cu*cu - 1.5*U2);
				
			}else if(simtype == 1){
				
				feq = w[l] * (rhofeq + 3.0*rhoinit*cu + 4.5*rhoinit*cu*cu -
				1.5*rhoinit*U2);
				
			}
			
			FMf = feq;
			
			index = ixyz + l*Nnx*Nny*Nnz;
			indexidjdkd = ixyzd + l*Nnx*Nny*Nnz;
			
			f[index]=FMb+(f[indexidjdkd]-FMf);
			
			l = 10; //********************
			
			rhofeq = 1.0;
			vxfeq = vx[ixyzd]; vyfeq = vy[ixyzd]; vzfeq = vz[ixyzd];
			
			cu = cx[l]*vxfeq + cy[l]*vyfeq + cz[l]*vzfeq;
			U2 = vxfeq*vxfeq + vyfeq*vyfeq + vzfeq*vzfeq;
			
			if(simtype == 0){
				
				feq = w[l] * rhofeq * (1.0 + 3.0*cu + 4.5*cu*cu - 1.5*U2);
				
			}else if(simtype == 1){
				
				feq = w[l] * (rhofeq + 3.0*rhoinit*cu + 4.5*rhoinit*cu*cu -
				1.5*rhoinit*U2);
				
			}
			
			FMb = feq;
			
			rhofeq = rho[ixyzd]; vxfeq = vx[ixyzd]; vyfeq = vy[ixyzd];
			vzfeq = vz[ixyzd];
			
			cu = cx[l]*vxfeq + cy[l]*vyfeq + cz[l]*vzfeq;
			U2 = vxfeq*vxfeq + vyfeq*vyfeq + vzfeq*vzfeq;
			
			if(simtype == 0){
				
				feq = w[l] * rhofeq * (1.0 + 3.0*cu + 4.5*cu*cu - 1.5*U2);
				
			}else if(simtype == 1){
				
				feq = w[l] * (rhofeq + 3.0*rhoinit*cu + 4.5*rhoinit*cu*cu -
				1.5*rhoinit*U2);
				
			}
			
			FMf = feq;
			
			index = ixyz + l*Nnx*Nny*Nnz;
			indexidjdkd = ixyzd + l*Nnx*Nny*Nnz;
			
			f[index]=FMb+(f[indexidjdkd]-FMf);
			
			l = 14; //********************
			
			rhofeq = 1.0;
			vxfeq = vx[ixyzd]; vyfeq = vy[ixyzd]; vzfeq = vz[ixyzd];
			
			cu = cx[l]*vxfeq + cy[l]*vyfeq + cz[l]*vzfeq;
			U2 = vxfeq*vxfeq + vyfeq*vyfeq + vzfeq*vzfeq;
			
			if(simtype == 0){
				
				feq = w[l] * rhofeq * (1.0 + 3.0*cu + 4.5*cu*cu - 1.5*U2);
				
			}else if(simtype == 1){
				
				feq = w[l] * (rhofeq + 3.0*rhoinit*cu + 4.5*rhoinit*cu*cu -
				1.5*rhoinit*U2);
				
			}
			
			FMb = feq;
			
			rhofeq = rho[ixyzd]; vxfeq = vx[ixyzd]; vyfeq = vy[ixyzd];
			vzfeq = vz[ixyzd];
			
			cu = cx[l]*vxfeq + cy[l]*vyfeq + cz[l]*vzfeq;
			U2 = vxfeq*vxfeq + vyfeq*vyfeq + vzfeq*vzfeq;
			
			if(simtype == 0){
				
				feq = w[l] * rhofeq * (1.0 + 3.0*cu + 4.5*cu*cu - 1.5*U2);
				
			}else if(simtype == 1){
				
				feq = w[l] * (rhofeq + 3.0*rhoinit*cu + 4.5*rhoinit*cu*cu -
				1.5*rhoinit*U2);
				
			}
			
			FMf = feq;
			
			index = ixyz + l*Nnx*Nny*Nnz;
			indexidjdkd = ixyzd + l*Nnx*Nny*Nnz;
			
			f[index]=FMb+(f[indexidjdkd]-FMf);
			
			l = 16; //********************
			
			rhofeq = 1.0;
			vxfeq = vx[ixyzd]; vyfeq = vy[ixyzd]; vzfeq = vz[ixyzd];
			
			cu = cx[l]*vxfeq + cy[l]*vyfeq + cz[l]*vzfeq;
			U2 = vxfeq*vxfeq + vyfeq*vyfeq + vzfeq*vzfeq;
			
			if(simtype == 0){
				
				feq = w[l] * rhofeq * (1.0 + 3.0*cu + 4.5*cu*cu - 1.5*U2);
				
			}else if(simtype == 1){
				
				feq = w[l] * (rhofeq + 3.0*rhoinit*cu + 4.5*rhoinit*cu*cu -
				1.5*rhoinit*U2);
				
			}
			
			FMb = feq;
			
			rhofeq = rho[ixyzd]; vxfeq = vx[ixyzd]; vyfeq = vy[ixyzd];
			vzfeq = vz[ixyzd];
			
			cu = cx[l]*vxfeq + cy[l]*vyfeq + cz[l]*vzfeq;
			U2 = vxfeq*vxfeq + vyfeq*vyfeq + vzfeq*vzfeq;
			
			if(simtype == 0){
				
				feq = w[l] * rhofeq * (1.0 + 3.0*cu + 4.5*cu*cu - 1.5*U2);
				
			}else if(simtype == 1){
				
				feq = w[l] * (rhofeq + 3.0*rhoinit*cu + 4.5*rhoinit*cu*cu -
				1.5*rhoinit*U2);
				
			}
			
			FMf = feq;
			
			index = ixyz + l*Nnx*Nny*Nnz;
			indexidjdkd = ixyzd + l*Nnx*Nny*Nnz;
			
			f[index]=FMb+(f[indexidjdkd]-FMf);
			
		}
	}
}

/// Zou-He boundary condition for LB nodes on left wall - specifies velocities
/// and calculates density

template<typename VALUE_TYPE, typename INDEX_TYPE>
__global__ void zouHeLeft(
	VALUE_TYPE *f,
	VALUE_TYPE latvxleft,
	VALUE_TYPE latvyleft,
	VALUE_TYPE latvzleft,
	INDEX_TYPE Nnx,
	INDEX_TYPE Nny,
	INDEX_TYPE Nnz,
	VALUE_TYPE *rho,
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
		
		VALUE_TYPE rhozh = 0.0, vxzh = 0.0, vyzh = 0.0, vzzh = 0.0;
		VALUE_TYPE Nyxleft, Nzxleft;
		
		// Left wall
		if(i == 0){
			
			vx[ixyz] = latvxleft;
			vy[ixyz] = latvyleft;
			vz[ixyz] = latvzleft;
			
			if(simtype == 0){
				
				rho[ixyz] = (f[i0]+f[i3]+f[i4]+f[i5]+f[i6]+f[i11]+f[i12]+f[i17]
							+f[i18]+2*(f[i2]+f[i8]+f[i10]+f[i14]+f[i16]))
							/(1-vx[ixyz]);
							
				Nyxleft = (f[i3]+f[i11]+f[i17]-f[i4]-f[i12]-f[i18])/2.0-
							(1.0/3.0)*rho[ixyz]*vy[ixyz];
				Nzxleft = (f[i5]+f[i11]+f[i18]-f[i6]-f[i12]-f[i17])/2.0-
							(1.0/3.0)*rho[ixyz]*vz[ixyz];
							
				rhozh = rho[ixyz];
				vxzh = vx[ixyz]; vyzh = vy[ixyz]; vzzh = vz[ixyz];
				
				f[i1] =  f[i2]  + (1.0/3.0)*rhozh*vxzh;
				f[i7] =  f[i8]  + (1.0/6.0)*rhozh*(vxzh + vyzh) - Nyxleft;
				f[i9] =  f[i10] + (1.0/6.0)*rhozh*(vxzh + vzzh) - Nzxleft;
				f[i13] = f[i14] - (1.0/6.0)*rhozh*(vyzh - vxzh) + Nyxleft;
				f[i15] = f[i16] - (1.0/6.0)*rhozh*(vzzh - vxzh) + Nzxleft;
				
			}else if(simtype == 1){
				
				rho[ixyz] = f[i0]+f[i3]+f[i4]+f[i5]+f[i6]+f[i11]+f[i12]+f[i17]
							+f[i18]+2*(f[i2]+f[i8]+f[i10]+f[i14]+f[i16])
							+vx[ixyz];
				
				Nyxleft = (f[i3]+f[i11]+f[i17]-f[i4]-f[i12]-f[i18])/2.0-
							(1.0/3.0)*vy[ixyz];
				Nzxleft = (f[i5]+f[i11]+f[i18]-f[i6]-f[i12]-f[i17])/2.0-
							(1.0/3.0)*vz[ixyz];
				
				vxzh = vx[ixyz]; vyzh = vy[ixyz]; vzzh = vz[ixyz];
				
				f[i1] =  f[i2]  + (1.0/3.0)*vxzh;
				f[i7] =  f[i8]  + (1.0/6.0)*(vxzh + vyzh) - Nyxleft;
				f[i9] =  f[i10] + (1.0/6.0)*(vxzh + vzzh) - Nzxleft;
				f[i13] = f[i14] - (1.0/6.0)*(vyzh - vxzh) + Nyxleft;
				f[i15] = f[i16] - (1.0/6.0)*(vzzh - vxzh) + Nzxleft;
				
			}
		}
	}	
}

/// Zou-He boundary condition for LB nodes on right wall - uses zero-gradient
/// density and velocities

template<typename VALUE_TYPE, typename INDEX_TYPE>
__global__ void zouHeRight(
	VALUE_TYPE *f,
	VALUE_TYPE latvxright,
	VALUE_TYPE latvyright,
	VALUE_TYPE latvzright,
	INDEX_TYPE Nnx,
	INDEX_TYPE Nny,
	INDEX_TYPE Nnz,
	VALUE_TYPE *rho,
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
		
		VALUE_TYPE rhozh = 0.0, vxzh = 0.0, vyzh = 0.0, vzzh = 0.0;
		VALUE_TYPE Nyxright, Nzxright;
		
		INDEX_TYPE kd, jd, id;
		INDEX_TYPE ixyzd;
		
		// Right wall
		if(i == Nnx-1){
			
			kd = k; jd = j; id = Nnx-2;
			ixyzd = id + jd*Nnx + kd*Nnx*Nny;
			
			if(simtype == 0){
				// Needs to be checked to make sure works - works!
				rho[ixyz] = rho[ixyzd];
				vx[ixyz] = (f[i0]+f[i3]+f[i4]+f[i5]+f[i6]+f[i11]+f[i12]+
							f[i17]+f[i18]+
							2*(f[i1]+f[i7]+f[i9]+f[i13]+f[i15]))
						    /rho[ixyz]-1;
				vy[ixyz] = vy[ixyzd];
				vz[ixyz] = vz[ixyzd];
				
				Nyxright = (f[i3]+f[i11]+f[i17]-f[i4]-f[i12]-f[i18])/2.0-
							(1.0/3.0)*rho[ixyz]*vy[ixyz];
				Nzxright = (f[i5]+f[i11]+f[i18]-f[i6]-f[i12]-f[i17])/2.0-
							(1.0/3.0)*rho[ixyz]*vz[ixyz];
				
				rhozh = rho[ixyz];
				vxzh = vx[ixyz]; vyzh = vy[ixyz]; vzzh = vz[ixyz];
				
				f[i2] =  f[i1]  - (1.0/3.0)*rhozh*vxzh;
				f[i8] =  f[i7]  - (1.0/6.0)*rhozh*(vxzh + vyzh) + Nyxright;
				f[i10] = f[i9]  - (1.0/6.0)*rhozh*(vxzh + vzzh) + Nzxright;
				f[i14] = f[i13] + (1.0/6.0)*rhozh*(vyzh - vxzh) - Nyxright;
				f[i16] = f[i15] + (1.0/6.0)*rhozh*(vzzh - vxzh) - Nzxright;
				
			}else if(simtype == 1){
				
				rho[ixyz] = rho[ixyzd];
				vx[ixyz] = f[i0]+f[i3]+f[i4]+f[i5]+f[i6]+f[i11]+f[i12]+
						   f[i17]+f[i18]+
						   2*(f[i1]+f[i7]+f[i9]+f[i13]+f[i15])
						   -rho[ixyz];
				vy[ixyz] = vy[ixyzd];
				vz[ixyz] = vz[ixyzd];
				
				Nyxright = (f[i3]+f[i11]+f[i17]-f[i4]-f[i12]-f[i18])/2.0-
							(1.0/3.0)*vy[ixyz];
				Nzxright = (f[i5]+f[i11]+f[i18]-f[i6]-f[i12]-f[i17])/2.0-
							(1.0/3.0)*vz[ixyz];
				
				vxzh = vx[ixyz];  vyzh = vy[ixyz];  vzzh = vz[ixyz];
				
				f[i2] =  f[i1]  - (1.0/3.0)*vxzh;
				f[i8] =  f[i7]  - (1.0/6.0)*(vxzh + vyzh) + Nyxright;
				f[i10] = f[i9]  - (1.0/6.0)*(vxzh + vzzh) + Nzxright;
				f[i14] = f[i13] + (1.0/6.0)*(vyzh - vxzh) - Nyxright;
				f[i16] = f[i15] + (1.0/6.0)*(vzzh - vxzh) - Nzxright;
				
			}
		}
	}	
}
