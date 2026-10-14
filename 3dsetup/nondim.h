#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <iomanip>
#include <sys/time.h>

#include "../3dsetup/typedefs.h"

/// Defines paramters used in the simulations

template<typename VALUE_TYPE, typename INDEX_TYPE>
void paramCalc(
	VALUE_TYPE &convfactpresscp,
	VALUE_TYPE &convfactrhocrho,
	VALUE_TYPE &convfactvelcv,
	INDEX_TYPE &delxdim,
	std::string &filenameinputdata,
	std::ofstream &finputdata,
	VALUE_TYPE &latvisc,
	VALUE_TYPE &latvxback,
	VALUE_TYPE &latvxbottom,
	VALUE_TYPE &latvxfront,
	VALUE_TYPE &latvxleft,
	VALUE_TYPE &latvxright,
	VALUE_TYPE &latvxtop,
	VALUE_TYPE &latvyback,
	VALUE_TYPE &latvybottom,
	VALUE_TYPE &latvyfront,
	VALUE_TYPE &latvyleft,
	VALUE_TYPE &latvyright,
	VALUE_TYPE &latvytop,
	VALUE_TYPE &latvzback,
	VALUE_TYPE &latvzbottom,
	VALUE_TYPE &latvzfront,
	VALUE_TYPE &latvzleft,
	VALUE_TYPE &latvzright,
	VALUE_TYPE &latvztop,
	VALUE_TYPE &latre,
	INDEX_TYPE &Ncx,
	INDEX_TYPE &Ncy,
	INDEX_TYPE &Ncz,
	VALUE_TYPE &numoflatupdates,
	INDEX_TYPE &Nnx,
    INDEX_TYPE &Nny,
    INDEX_TYPE &Nnz,
    VALUE_TYPE &physchardim,
    VALUE_TYPE &physcharvel,
	VALUE_TYPE &physdelxcl,
	VALUE_TYPE &physdeltct,
	VALUE_TYPE &physlengthx,
	VALUE_TYPE &physlengthy,
	VALUE_TYPE &physlengthz,
	VALUE_TYPE &physrho,
	VALUE_TYPE &physvisc,
	VALUE_TYPE &physvxback,
	VALUE_TYPE &physvxbottom,
	VALUE_TYPE &physvxfront,
	VALUE_TYPE &physvxleft,
	VALUE_TYPE &physvxright,
	VALUE_TYPE &physvxtop,
	VALUE_TYPE &physvyback,
	VALUE_TYPE &physvybottom,
	VALUE_TYPE &physvyfront,
	VALUE_TYPE &physvyleft,
	VALUE_TYPE &physvyright,
	VALUE_TYPE &physvytop,
	VALUE_TYPE &physvzback,
	VALUE_TYPE &physvzbottom,
	VALUE_TYPE &physvzfront,
	VALUE_TYPE &physvzleft,
	VALUE_TYPE &physvzright,
	VALUE_TYPE &physvztop,
	INDEX_TYPE &relaxtype,
	VALUE_TYPE &rhoinit,
	VALUE_TYPE &tau,
	VALUE_TYPE &timestepsfor1s,
	VALUE_TYPE &vxinit,
	VALUE_TYPE &vyinit,
	VALUE_TYPE &vzinit){

	// Calculates the lattice density and sets the intial density
	rhoinit = physrho/convfactrhocrho;
	
	// Calculates lattice viscosity
	latvisc = (tau-0.5)/3;
	
	// Calculates physical space increment
	if(delxdim == 0){
		physdelxcl = physlengthx/Ncx;
	}else if(delxdim == 1){
		physdelxcl = physlengthy/Ncy;
	}else if(delxdim == 2){
		physdelxcl = physlengthz/Ncz;
	}
	// Calculates physical time step
	physdeltct = ((tau-0.5)/3)*((pow(physdelxcl,2)/physvisc));
	
	// Calculates velocity conversion factor
	convfactvelcv = physdelxcl/physdeltct;
	
	// Calculates pressure conversion factor
	// The density conversion factor in the eq is neglected as it is 1
	convfactpresscp = pow(convfactvelcv,2);
	
	// Calculates lattice velocities
	latvxback =   physvxback/convfactvelcv;
	latvxbottom = physvxbottom/convfactvelcv;
	latvxfront =  physvxfront/convfactvelcv;
	latvxleft =   physvxleft/convfactvelcv;
	latvxright =  physvxright/convfactvelcv;
	latvxtop =    physvxtop/convfactvelcv;
	
	latvyback =   physvyback/convfactvelcv;
	latvybottom = physvybottom/convfactvelcv;
	latvyfront =  physvyfront/convfactvelcv;
	latvyleft =   physvyleft/convfactvelcv;
	latvyright =  physvyright/convfactvelcv;
	latvytop =    physvytop/convfactvelcv;
	
	latvzback =   physvzback/convfactvelcv;
	latvzbottom = physvzbottom/convfactvelcv;
	latvzfront =  physvzfront/convfactvelcv;
	latvzleft =   physvzleft/convfactvelcv;
	latvzright =  physvzright/convfactvelcv;
	latvztop =    physvztop/convfactvelcv;
	
	// Calculates latre
	VALUE_TYPE latcharvel = physcharvel/convfactvelcv;
	VALUE_TYPE latchardim = physchardim/physdelxcl;
	latre = (latcharvel*latchardim)/latvisc;
	
	// Sets initial velocities at all nodes
	vxinit = physvxleft/convfactvelcv; vyinit = 0.0; vzinit = 0.0;
	
	// Calculates number of timesteps required for one second of real time
	timestepsfor1s = 1.0/physdeltct;
	
	// Calculates number of lattice updates required for one second
	numoflatupdates = Nnx*Nny*Nnz*timestepsfor1s;
	
	// Following not calculated but useful output
	std::cout<<"Physical length in - x: "<<physlengthx<<
	", y: "<<physlengthy<<
	", z: "<<physlengthz<<
	std::endl;
	
	std::cout<<"Number of cells in - x: "<<Ncx<<
	", y: "<<Ncy<<
	", z: "<<Ncz<<
	std::endl;
	
	std::cout<<"Physical density: "<<physrho<<std::endl;
	std::cout<<"Density conversion factor: "<<convfactrhocrho<<std::endl;
	std::cout<<"Physical viscosity: "<<physvisc<<std::endl;
	
	std::cout<<"Physical vx - back: "<<  physvxback<<
	", bottom: "<<					physvxbottom<<
	", front: "<<					physvxfront<<
	", left: "<<					physvxleft<<
	", right: "<<					physvxright<<
	", top: "<<						physvxtop<<
	std::endl;
	
	std::cout<<"Physical vy - back: "<<  physvyback<<
	", bottom: "<<					physvybottom<<
	", front: "<<					physvyfront<<
	", left: "<<					physvyleft<<
	", right: "<<					physvyright<<
	", top: "<<						physvytop<<
	std::endl;
	
	std::cout<<"Physical vz - back: "<<  physvzback<<
	", bottom: "<<					physvzbottom<<
	", front: "<<					physvzfront<<
	", left: "<<					physvzleft<<
	", right: "<<					physvzright<<
	", top: "<<						physvztop<<
	std::endl;
	
	std::cout<<"Relaxation factor: "<<tau<<std::endl;
	std::cout<<"Relaxation type: "<<relaxtype<<std::endl;
	
	std::cout<<"Lattice density/initial density: "<<rhoinit<<std::endl;
	std::cout<<"Lattice viscosity: "<<latvisc<<std::endl;
	std::cout<<"Physical delta x: "<<physdelxcl<<std::endl;
	std::cout<<"Physical delta t: "<<physdeltct<<std::endl;
	std::cout<<"Velocity conversion factor: "<<convfactvelcv<<std::endl;
	std::cout<<"Pressure conversion factor: "<<convfactpresscp<<std::endl;
	
	std::cout<<"Lattice vx - back: "<<   latvxback<<
	", bottom: "<<					latvxbottom<<
	", front: "<<					latvxfront<<
	", left: "<<					latvxleft<<
	", right: "<<					latvxright<<
	", top: "<<						latvxtop<<
	std::endl;
	
	std::cout<<"Lattice vy - back: "<<   latvyback<<
	", bottom: "<<					latvybottom<<
	", front: "<<					latvyfront<<
	", left: "<<					latvyleft<<
	", right: "<<					latvyright<<
	", top: "<<						latvytop<<
	std::endl;
	
	std::cout<<"Lattice vz - back: "<<   latvzback<<
	", bottom: "<<					latvzbottom<<
	", front: "<<					latvzfront<<
	", left: "<<					latvzleft<<
	", right: "<<					latvzright<<
	", top: "<<						latvztop<<
	std::endl;
	
	std::cout<<"Initial velocities for nodes in - x: "<<vxinit<<
	", y: "<<vyinit<<
	", z: "<<vzinit<<
	std::endl;
	
	std::cout<<"Lattice Reynolds number: "<<latre<<std::endl;
	std::cout<<"Time steps required for one second: "<<timestepsfor1s<<std::endl;
	std::cout<<"No. of lattice updates: "<<numoflatupdates<<std::endl;
	
	finputdata.open(filenameinputdata,std::ios::out | std::ios::app);
	if(finputdata.is_open()){
		finputdata<<"Physical length in - x: "<<physlengthx<<
		", y: "<<physlengthy<<
		", z: "<<physlengthz<<
		std::endl;
		
		finputdata<<"Number of cells in - x: "<<Ncx<<
		", y: "<<Ncy<<
		", z: "<<Ncz<<
		std::endl;
		
		finputdata<<"Physical density: "<<physrho<<std::endl;
		finputdata<<"Density conversion factor: "<<convfactrhocrho<<std::endl;
		finputdata<<"Physical viscosity: "<<physvisc<<std::endl;
		
		finputdata<<"Physical vx - back: "<<  physvxback<<
		", bottom: "<<					physvxbottom<<
		", front: "<<					physvxfront<<
		", left: "<<					physvxleft<<
		", right: "<<					physvxright<<
		", top: "<<						physvxtop<<
		std::endl;
		
		finputdata<<"Physical vy - back: "<<  physvyback<<
		", bottom: "<<					physvybottom<<
		", front: "<<					physvyfront<<
		", left: "<<					physvyleft<<
		", right: "<<					physvyright<<
		", top: "<<						physvytop<<
		std::endl;
		
		finputdata<<"Physical vz - back: "<<  physvzback<<
		", bottom: "<<					physvzbottom<<
		", front: "<<					physvzfront<<
		", left: "<<					physvzleft<<
		", right: "<<					physvzright<<
		", top: "<<						physvztop<<
		std::endl;
		
		finputdata<<"Relaxation factor: "<<tau<<std::endl;
		finputdata<<"Relaxation type: "<<relaxtype<<std::endl;
		finputdata<<"Lattice density/initial density: "<<rhoinit<<std::endl;
		finputdata<<"Lattice viscosity: "<<latvisc<<std::endl;
		finputdata<<"Physical delta x: "<<physdelxcl<<std::endl;
		finputdata<<"Physical delta t: "<<physdeltct<<std::endl;
		finputdata<<"Velocity conversion factor: "<<convfactvelcv<<std::endl;
		finputdata<<"Pressure conversion factor: "<<convfactpresscp<<std::endl;
		
		finputdata<<"Lattice vx - back: "<<   latvxback<<
		", bottom: "<<					latvxbottom<<
		", front: "<<					latvxfront<<
		", left: "<<					latvxleft<<
		", right: "<<					latvxright<<
		", top: "<<						latvxtop<<
		std::endl;
		
		finputdata<<"Lattice vy - back: "<<   latvyback<<
		", bottom: "<<					latvybottom<<
		", front: "<<					latvyfront<<
		", left: "<<					latvyleft<<
		", right: "<<					latvyright<<
		", top: "<<						latvytop<<
		std::endl;
		
		finputdata<<"Lattice vz - back: "<<   latvzback<<
		", bottom: "<<					latvzbottom<<
		", front: "<<					latvzfront<<
		", left: "<<					latvzleft<<
		", right: "<<					latvzright<<
		", top: "<<						latvztop<<
		std::endl;
		
		finputdata<<"Initial velocities for nodes in - x: "<<vxinit<<
		", y: "<<vyinit<<
		", z: "<<vzinit<<
		std::endl;
		
		finputdata<<"Lattice Reynolds number: "<<latre<<std::endl;
		finputdata<<"Time steps required for one second: "<<timestepsfor1s<<
		std::endl;
		finputdata<<"No. of lattice updates: "<<numoflatupdates<<std::endl;
		
		finputdata.close();
		
	}else{
		
		std::cout<<"Unable to open file! - inputdata"<<std::endl;
		
	}
}
