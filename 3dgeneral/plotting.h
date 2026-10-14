#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <iomanip>
#include <sys/time.h>
#include <algorithm>

#include "../3dbase/constants.h"
#include "../3dsetup/typedefs.h"
#include "../cudafunc/cudaatomic.h"

/// Sets up linkwise pressure Tecplot file and outputs pressures

template<typename VALUE_TYPE, typename INDEX_TYPE>
void tecplot3DLinkwisePressure(
	VALUE_TYPE &convfactpresscp,
	std::string &filename3dpressure,
	std::ofstream &f3dp,
	INDEX_TYPE &Ncx,
    INDEX_TYPE &Ncy,
    INDEX_TYPE &Ncz,
	INDEX_TYPE &Nnx,
	INDEX_TYPE &Nny,
	INDEX_TYPE &Nnz,
	INDEX_TYPE &periodicinx,
	INDEX_TYPE &periodiciny,
	INDEX_TYPE &periodicinz,
	VALUE_TYPE &physlengthx,
	VALUE_TYPE &physlengthy, 
	VALUE_TYPE &physlengthz,
	VALUE_TYPE *rho,
	INDEX_TYPE &step){
	
	std::string fname;
	fname = filename3dpressure;
	f3dp.open(fname.c_str(), std::ios::out | std::ios::app);
	
	INDEX_TYPE ixyz;
    INDEX_TYPE cellsx = Ncx, cellsy = Ncy, cellsz = Ncz;
	
	INDEX_TYPE nodesinx, startnodex, endnodex;
	INDEX_TYPE isnotperiodicmodx;
    if(periodicinx == 0){
		
		nodesinx = Nnx;
		startnodex = 0;
		endnodex = Nnx-1;
		isnotperiodicmodx = 1;
		
	}else if(periodicinx == 1){
		
		nodesinx = Nnx-2;
		startnodex = 1;
		endnodex = Nnx-2;
		isnotperiodicmodx = 0;
		
	}
	
	INDEX_TYPE nodesiny, startnodey, endnodey;
	INDEX_TYPE isnotperiodicmody;
    if(periodiciny == 0){
		
		nodesiny = Nny;
		startnodey = 0;
		endnodey = Nny-1;
		isnotperiodicmody = 1;
		
	}else if(periodiciny == 1){
		
		nodesiny = Nny-2;
		startnodey = 1;
		endnodey = Nny-2;
		isnotperiodicmody = 0;
		
	}
	
	INDEX_TYPE nodesinz, startnodez, endnodez;
	INDEX_TYPE isnotperiodicmodz;
    if(periodicinz == 0){
		
		nodesinz = Nnz;
		startnodez = 0;
		endnodez = Nnz-1;
		isnotperiodicmodz = 1;
		
	}else if(periodicinz == 1){
		
		nodesinz = Nnz-2;
		startnodez = 1;
		endnodez = Nnz-2;
		isnotperiodicmodz = 0;
		
	}
	
	if(f3dp.is_open()){
		
		if(step == 0){
			
			f3dp<<"TITLE=\"Pressure across entire 3D domain\""<<"\n";
			
			f3dp<<"VARIABLES="<<
			"\"Channel length (m)\""<<","<<
			"\"Channel height (m)\""<<","<<
			"\"Channel breadth (m)\""<<","<<
			"\"Pressure (Pa)\""<<"\n";
		
		}
		
		f3dp.close();
		
	} else std::cout << "Unable to open pressure output file!"<<std::endl;
	
	fname = filename3dpressure;
	f3dp.open(fname.c_str(), std::ios::out | std::ios::app);
	
	f3dp<<"\nZONE T="<<
	"\"Time step "<<step<<"\""<<
	", i="<<nodesinx<<
	", j="<<nodesiny<<
	", k="<<nodesinz<<
	", DATAPACKING=POINT,ZONETYPE=ORDERED";
	
	// Arbitrary reference pressure of 0 Pa chosen
	VALUE_TYPE nondimrefpressure = 0.0/convfactpresscp;
	VALUE_TYPE nondimpressurechange;
	
	for(INDEX_TYPE k = startnodez; k <= endnodez; k++){
		for(INDEX_TYPE j = startnodey; j <= endnodey; j++){
			for(INDEX_TYPE i = startnodex; i <= endnodex; i++){
				
					ixyz = i + j*Nnx + k*Nnx*Nny;
					
					// Reference density is 1
					nondimpressurechange = (rho[ixyz]-1.0)*(1.0/3.0);
					
					f3dp<<"\n"
					<<std::setprecision(15)
					<<(VALUE_TYPE(i+isnotperiodicmodx)-0.5)*
					(physlengthx/VALUE_TYPE(cellsx))
					<<" "
					<<(VALUE_TYPE(j+isnotperiodicmody)-0.5)*
					(physlengthy/VALUE_TYPE(cellsy))
					<<" "
					<<(VALUE_TYPE(k+isnotperiodicmodz)-0.5)*
					(physlengthz/VALUE_TYPE(cellsz))
					<<" "
					<<(nondimpressurechange+nondimrefpressure)*convfactpresscp;
					
			}
		}
	}
	
	f3dp.close();
	
}

/// Sets up linkwise velocity Tecplot file and outputs velocities

template<typename VALUE_TYPE, typename INDEX_TYPE>
void tecplot3DLinkwiseVelocities(
	VALUE_TYPE &convfactvelcv,
	std::string &filename3dvelocities,
	std::ofstream &f3dv,
	INDEX_TYPE &Ncx,
    INDEX_TYPE &Ncy,
    INDEX_TYPE &Ncz,
	INDEX_TYPE &Nnx,
	INDEX_TYPE &Nny,
	INDEX_TYPE &Nnz,
	INDEX_TYPE &periodicinx,
	INDEX_TYPE &periodiciny,
	INDEX_TYPE &periodicinz,
	VALUE_TYPE &physlengthx,
	VALUE_TYPE &physlengthy, 
	VALUE_TYPE &physlengthz,
	INDEX_TYPE step,
	VALUE_TYPE *vx,
	VALUE_TYPE *vy,
	VALUE_TYPE *vz){
	
	std::string fname;
	fname = filename3dvelocities;
	f3dv.open(fname.c_str(), std::ios::out | std::ios::app);
	
	INDEX_TYPE ixyz;
    INDEX_TYPE cellsx = Ncx, cellsy = Ncy, cellsz = Ncz;
	
	INDEX_TYPE nodesinx, startnodex, endnodex;
	INDEX_TYPE isnotperiodicmodx;
    if(periodicinx == 0){
		
		nodesinx = Nnx;
		startnodex = 0;
		endnodex = Nnx-1;
		isnotperiodicmodx = 1;
		
	}else if(periodicinx == 1){
		
		nodesinx = Nnx-2;
		startnodex = 1;
		endnodex = Nnx-2;
		isnotperiodicmodx = 0;
		
	}
	
	INDEX_TYPE nodesiny, startnodey, endnodey;
	INDEX_TYPE isnotperiodicmody;
    if(periodiciny == 0){
		
		nodesiny = Nny;
		startnodey = 0;
		endnodey = Nny-1;
		isnotperiodicmody = 1;
		
	}else if(periodiciny == 1){
		
		nodesiny = Nny-2;
		startnodey = 1;
		endnodey = Nny-2;
		isnotperiodicmody = 0;
		
	}
	
	INDEX_TYPE nodesinz, startnodez, endnodez;
	INDEX_TYPE isnotperiodicmodz;
    if(periodicinz == 0){
		
		nodesinz = Nnz;
		startnodez = 0;
		endnodez = Nnz-1;
		isnotperiodicmodz = 1;
		
	}else if(periodicinz == 1){
		
		nodesinz = Nnz-2;
		startnodez = 1;
		endnodez = Nnz-2;
		isnotperiodicmodz = 0;
		
	}
	
	if(f3dv.is_open()){
		
		if(step == 0){
			
			f3dv<<"TITLE=\"Velocities across entire 3D domain\""<<"\n";
			
			f3dv<<"VARIABLES="<<
			"\"Channel length (m)\""<<","<<
			"\"Channel height (m)\""<<","<<
			"\"Channel breadth (m)\""<<","<<
			"\"x velocity (m/s)\""<<","<<
			"\"y velocity (m/s)\""<<","<<
			"\"z velocity (m/s)\""<<"\n";
			
		}
		
		f3dv.close();
		
	} else std::cout << "Unable to open velocity output file!"<<std::endl;
	
	fname = filename3dvelocities;
	f3dv.open(fname.c_str(), std::ios::out | std::ios::app);
	
	f3dv<<"\nZONE T="<<
	"\"Time step "<<step<<"\""<<
	", i="<<nodesinx<<
	", j="<<nodesiny<<
	", k="<<nodesinz<<
	", DATAPACKING=POINT,ZONETYPE=ORDERED";
	
	for(INDEX_TYPE k = startnodez; k <= endnodez; k++){
		for(INDEX_TYPE j = startnodey; j <= endnodey; j++){
			for(INDEX_TYPE i = startnodex; i <= endnodex; i++){
				
					ixyz = i + j*Nnx + k*Nnx*Nny;
					
					f3dv<<"\n"
					<<std::setprecision(15)
					<<(VALUE_TYPE(i+isnotperiodicmodx)-0.5)*
					(physlengthx/VALUE_TYPE(cellsx))
					<<" "
					<<(VALUE_TYPE(j+isnotperiodicmody)-0.5)*
					(physlengthy/VALUE_TYPE(cellsy))
					<<" "
					<<(VALUE_TYPE(k+isnotperiodicmodz)-0.5)*
					(physlengthz/VALUE_TYPE(cellsz))
					<<" "
					<<vx[ixyz]*convfactvelcv
					<<" "
					<<vy[ixyz]*convfactvelcv
					<<" "
					<<vz[ixyz]*convfactvelcv;
					
			}
		}
	}
	
	f3dv.close();
	
}

/// Sets up linkwise Plane Couette simulation x-velocity vs. analytical x-
/// velocity Tecplot file and outputs analytical and simulation x-
/// velocities

template<typename VALUE_TYPE, typename INDEX_TYPE>
void tecplot3DLinkwiseXvelocityVsAnalyticalPlaneCouette(
	VALUE_TYPE &convfactvelcv,
	std::string &filename3dxvelvsanalytical,
	std::ofstream &f3dxvelvsa,
	INDEX_TYPE &Ncx,
    INDEX_TYPE &Ncy,
    INDEX_TYPE &Ncz,
	INDEX_TYPE &Nnx,
	INDEX_TYPE &Nny,
	INDEX_TYPE &Nnz,
	INDEX_TYPE &periodicinx,
	INDEX_TYPE &periodiciny,
	INDEX_TYPE &periodicinz,
	VALUE_TYPE &physlengthx,
	VALUE_TYPE &physlengthy, 
	VALUE_TYPE &physlengthz,
	VALUE_TYPE &physvel,
	INDEX_TYPE &step,
	VALUE_TYPE *vx){
	
	std::string fname;
	fname = filename3dxvelvsanalytical;
    f3dxvelvsa.open(fname.c_str(), std::ios::out | std::ios::app);
	
	INDEX_TYPE ixyz;
    INDEX_TYPE cellsy = Ncy;
	
	INDEX_TYPE startnodex, endnodex;
    if(periodicinx == 0){
		
		startnodex = 0;
		endnodex = Nnx-1;
		
	}else if(periodicinx == 1){
		
		startnodex = 1;
		endnodex = Nnx-2;
		
	}
	
	INDEX_TYPE nodesiny, startnodey, endnodey;
	INDEX_TYPE isnotperiodicmody;
    if(periodiciny == 0){
		
		nodesiny = Nny;
		startnodey = 0;
		endnodey = Nny-1;
		isnotperiodicmody = 1;
		
	}else if(periodiciny == 1){
		
		nodesiny = Nny-2;
		startnodey = 1;
		endnodey = Nny-2;
		isnotperiodicmody = 0;
		
	}
	
	INDEX_TYPE startnodez, endnodez;
    if(periodicinz == 0){
		
		startnodez = 0;
		endnodez = Nnz-1;
		
	}else if(periodicinz == 1){
		
		startnodez = 1;
		endnodez = Nnz-2;
		
	}
	
    if(f3dxvelvsa.is_open()){
		
		if(step == 0){
			
			 f3dxvelvsa<<"TITLE=\"Channel x velocity along length\""<<"\n";
			
			f3dxvelvsa<<"VARIABLES="<<
			"\"Physical x velocity (m/s)\""<<","<<
			"\"Physical height of channel (m)\""<<"\n";
			
		}
		
		f3dxvelvsa.close();
		
    } else std::cout << "Unable to open x vel vs. analytical vel file!"<<std::endl;
     
    fname = filename3dxvelvsanalytical;
    f3dxvelvsa.open(fname.c_str(), std::ios::out | std::ios::app);
    
    if(step == 0){
		
		for(INDEX_TYPE k = startnodez; k <= endnodez; k++){
			for(INDEX_TYPE i = startnodex; i <= endnodex; i++){
				
				f3dxvelvsa<<"\nZONE T="<<
				"\"Analytical solution, z:"<<k<<", x:"<<i<<"\", i="<<nodesiny<<
				", DATAPACKING=POINT,ZONETYPE=ORDERED,DT=(DOUBLE,DOUBLE)";
				
				for(INDEX_TYPE j = startnodey; j <= endnodey; j++){
					
					f3dxvelvsa<<"\n"<<
					std::setprecision(15)<<
					(physvel*(VALUE_TYPE(j+isnotperiodicmody)-0.5)*
					(physlengthy/VALUE_TYPE(cellsy)))/physlengthy
					<<" "<<
					(VALUE_TYPE(j+isnotperiodicmody)-0.5)*
					(physlengthy/VALUE_TYPE(cellsy));
					
				}
			}
		}
		
	}
	
	
	for(INDEX_TYPE k = startnodez; k <= endnodez; k++){
		for(INDEX_TYPE i = startnodex; i <= endnodex; i++){
			
			f3dxvelvsa<<"\nZONE T="<<
			"\"Time step "<<step<<", z:"<<k<<", x:"<<i<<"\", i="<<nodesiny<<
			", DATAPACKING=POINT,ZONETYPE=ORDERED,DT=(DOUBLE,DOUBLE)";
			
			for(VALUE_TYPE j = startnodey; j <= endnodey; j++){
				
				ixyz = i + j*Nnx + k*Nnx*Nny;
				
				f3dxvelvsa<<"\n"<<
				std::setprecision(15)<<
				vx[ixyz]*convfactvelcv
				<<" "<<
				(VALUE_TYPE(j+isnotperiodicmody)-0.5)*
				(physlengthy/VALUE_TYPE(cellsy));
				
			}
		}
	}
	
    f3dxvelvsa.close();
    
}

/// Sets up wetnode pressure Tecplot file and outputs pressures

template<typename VALUE_TYPE, typename INDEX_TYPE>
void tecplot3DWetnodePressure(
	VALUE_TYPE &convfactpresscp,
	std::string &filename3dpressure,
	std::ofstream &f3dp,
	INDEX_TYPE &Ncx,
    INDEX_TYPE &Ncy,
    INDEX_TYPE &Ncz,
	INDEX_TYPE &Nnx,
	INDEX_TYPE &Nny,
	INDEX_TYPE &Nnz,
	INDEX_TYPE &periodicinx,
	INDEX_TYPE &periodiciny,
	INDEX_TYPE &periodicinz,
	VALUE_TYPE &physlengthx,
	VALUE_TYPE &physlengthy, 
	VALUE_TYPE &physlengthz,
	VALUE_TYPE *rho,
	INDEX_TYPE &step){
	
	std::string fname;
	fname = filename3dpressure;
	f3dp.open(fname.c_str(), std::ios::out | std::ios::app);
	
	INDEX_TYPE ixyz;
    INDEX_TYPE cellsx = Ncx, cellsy = Ncy, cellsz = Ncz;
	
	INDEX_TYPE nodesinx, startnodex, endnodex;
	INDEX_TYPE isperiodicmodx;
    if(periodicinx == 0){
		
		nodesinx = Nnx;
		startnodex = 0;
		endnodex = Nnx-1;
		isperiodicmodx = 0;
		
	}else if(periodicinx == 1){
		
		nodesinx = Nnx-2;
		startnodex = 1;
		endnodex = Nnx-2;
		isperiodicmodx = 1;
		
	}
	
	INDEX_TYPE nodesiny, startnodey, endnodey;
	INDEX_TYPE isperiodicmody;
    if(periodiciny == 0){
		
		nodesiny = Nny;
		startnodey = 0;
		endnodey = Nny-1;
		isperiodicmody = 0;
		
	}else if(periodiciny == 1){
		
		nodesiny = Nny-2;
		startnodey = 1;
		endnodey = Nny-2;
		isperiodicmody = 1;
		
	}
	
	INDEX_TYPE nodesinz, startnodez, endnodez;
	INDEX_TYPE isperiodicmodz;
    if(periodicinz == 0){
		
		nodesinz = Nnz;
		startnodez = 0;
		endnodez = Nnz-1;
		isperiodicmodz = 0;
		
	}else if(periodicinz == 1){
		
		nodesinz = Nnz-2;
		startnodez = 1;
		endnodez = Nnz-2;
		isperiodicmodz = 1;
		
	}
	
	if(f3dp.is_open()){
		
		if(step == 0){
			
			f3dp<<"TITLE=\"Pressure across entire 3D domain\""<<"\n";
			
			f3dp<<"VARIABLES="<<
			"\"Channel length (m)\""<<","<<
			"\"Channel height (m)\""<<","<<
			"\"Channel breadth (m)\""<<","<<
			"\"Pressure (Pa)\""<<"\n";
			
		}
		
		f3dp.close();
		
	} else std::cout << "Unable to open pressure output file!"<<std::endl;
	
	fname = filename3dpressure;
	f3dp.open(fname.c_str(), std::ios::out | std::ios::app);
	
	f3dp<<"\nZONE T="<<
	"\"Time step "<<step<<"\""<<
	", i="<<nodesinx<<
	", j="<<nodesiny<<
	", k="<<nodesinz<<
	", DATAPACKING=POINT,ZONETYPE=ORDERED";
	
	// Arbitrary reference pressure of 0 Pa chosen
	VALUE_TYPE nondimrefpressure = 0.0/convfactpresscp;
	VALUE_TYPE nondimpressurechange;
	
	for(INDEX_TYPE k = startnodez; k <= endnodez; k++){
		for(INDEX_TYPE j = startnodey; j <= endnodey; j++){
			for(INDEX_TYPE i = startnodex; i <= endnodex; i++){
				
					ixyz = i + j*Nnx + k*Nnx*Nny;
					
					// Reference density is 1
					nondimpressurechange = (rho[ixyz]-1.0)*(1.0/3.0);
					
					f3dp<<"\n"
					<<std::setprecision(15)
					<<VALUE_TYPE(i-isperiodicmodx)*
					(physlengthx/VALUE_TYPE(cellsx))
					<<" "
					<<VALUE_TYPE(j-isperiodicmody)*
					(physlengthy/VALUE_TYPE(cellsy))
					<<" "
					<<VALUE_TYPE(k-isperiodicmodz)*
					(physlengthz/VALUE_TYPE(cellsz))
					<<" "
					<<(nondimpressurechange+nondimrefpressure)*convfactpresscp;
					
			}
		}
	}
	
	f3dp.close();
	
}

/// Sets up wetnode velocity Tecplot file and outputs velocities

template<typename VALUE_TYPE, typename INDEX_TYPE>
void tecplot3DWetnodeVelocities(
	VALUE_TYPE &convfactvelcv,
	std::string &filename3dvelocities,
	std::ofstream &f3dv,
	INDEX_TYPE &Ncx,
    INDEX_TYPE &Ncy,
    INDEX_TYPE &Ncz,
	INDEX_TYPE &Nnx,
	INDEX_TYPE &Nny,
	INDEX_TYPE &Nnz,
	INDEX_TYPE &periodicinx,
	INDEX_TYPE &periodiciny,
	INDEX_TYPE &periodicinz,
	VALUE_TYPE &physlengthx,
	VALUE_TYPE &physlengthy, 
	VALUE_TYPE &physlengthz,
	INDEX_TYPE &step,
	VALUE_TYPE *vx,
	VALUE_TYPE *vy,
	VALUE_TYPE *vz){
	
	std::string fname;
	fname = filename3dvelocities;
	f3dv.open(fname.c_str(), std::ios::out | std::ios::app);
	
	INDEX_TYPE ixyz;
    INDEX_TYPE cellsx = Ncx, cellsy = Ncy, cellsz = Ncz;
	
	INDEX_TYPE nodesinx, startnodex, endnodex;
	INDEX_TYPE isperiodicmodx;
    if(periodicinx == 0){
		
		nodesinx = Nnx;
		startnodex = 0;
		endnodex = Nnx-1;
		isperiodicmodx = 0;
		
	}else if(periodicinx == 1){
		
		nodesinx = Nnx-2;
		startnodex = 1;
		endnodex = Nnx-2;
		isperiodicmodx = 1;
		
	}
	
	INDEX_TYPE nodesiny, startnodey, endnodey;
	INDEX_TYPE isperiodicmody;
    if(periodiciny == 0){
		
		nodesiny = Nny;
		startnodey = 0;
		endnodey = Nny-1;
		isperiodicmody = 0;
		
	}else if(periodiciny == 1){
		
		nodesiny = Nny-2;
		startnodey = 1;
		endnodey = Nny-2;
		isperiodicmody = 1;
		
	}
	
	INDEX_TYPE nodesinz, startnodez, endnodez;
	INDEX_TYPE isperiodicmodz;
    if(periodicinz == 0){
		
		nodesinz = Nnz;
		startnodez = 0;
		endnodez = Nnz-1;
		isperiodicmodz = 0;
		
	}else if(periodicinz == 1){
		
		nodesinz = Nnz-2;
		startnodez = 1;
		endnodez = Nnz-2;
		isperiodicmodz = 1;
		
	}
	
	if(f3dv.is_open()){
		
		if(step == 0){
			
			f3dv<<"TITLE=\"Velocities across entire 3D domain\""<<"\n";
			
			f3dv<<"VARIABLES="<<
			"\"Channel length (m)\""<<","<<
			"\"Channel height (m)\""<<","<<
			"\"Channel breadth (m)\""<<","<<
			"\"x velocity (m/s)\""<<","<<
			"\"y velocity (m/s)\""<<","<<
			"\"z velocity (m/s)\""<<"\n";
			
		}
		
		f3dv.close();
		
	} else std::cout << "Unable to open velocity output file!"<<std::endl;
	
	fname = filename3dvelocities;
	f3dv.open(fname.c_str(), std::ios::out | std::ios::app);
	
	f3dv<<"\nZONE T="<<
	"\"Time step "<<step<<"\""<<
	", i="<<nodesinx<<
	", j="<<nodesiny<<
	", k="<<nodesinz<<
	", DATAPACKING=POINT,ZONETYPE=ORDERED";
	
	for(INDEX_TYPE k = startnodez; k <= endnodez; k++){
		for(INDEX_TYPE j = startnodey; j <= endnodey; j++){
			for(INDEX_TYPE i = startnodex; i <= endnodex; i++){
				
					ixyz = i + j*Nnx + k*Nnx*Nny;
					
					f3dv<<"\n"
					<<std::setprecision(15)
					<<VALUE_TYPE(i-isperiodicmodx)*
					(physlengthx/VALUE_TYPE(cellsx))
					<<" "
					<<VALUE_TYPE(j-isperiodicmody)*
					(physlengthy/VALUE_TYPE(cellsy))
					<<" "
					<<VALUE_TYPE(k-isperiodicmodz)*
					(physlengthz/VALUE_TYPE(cellsz))
					<<" "
					<<vx[ixyz]*convfactvelcv
					<<" "
					<<vy[ixyz]*convfactvelcv
					<<" "
					<<vz[ixyz]*convfactvelcv;
			}
		}
	}
	
	f3dv.close();
	
}
