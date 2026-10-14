#pragma once

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
#include "../3dgeneral/fileops.h"
#include "../3dsetup/typedefs.h"

template<typename VALUE_TYPE, typename INDEX_TYPE>
void filenameInputData(
	std::string &filenameinputdata,
	INDEX_TYPE &countsearch,
	INDEX_TYPE &restartsiminput,
	INDEX_TYPE remoteoutput,
	std::string &remoteoutputfolderichecint,
	std::string &remoteoutputfolderichecbatch,
	std::string &remoteoutputfolderaltint,
	std::string &remoteoutputfolderaltbatch,
	INDEX_TYPE &runonichec,
	std::string &foldername,
	std::string &codename,
	std::string &testcasename,
	INDEX_TYPE &simnum,
	INDEX_TYPE remotebatch){
	
	if(restartsiminput == 1){
	
		if(remoteoutput == 0){
			filenameinputdata = "../../local_output/" +
			foldername + "/" + codename + "/" + testcasename + std::to_string(simnum) +
			"_inputdata.dat";
		}else if (remoteoutput == 1){
			if (runonichec == 1) {
				if (remotebatch == 0) {
					filenameinputdata = remoteoutputfolderichecint +
						foldername + "/" + codename + "/" + testcasename + std::to_string(simnum) +
						"_inputdata.dat";
				}
				else if (remotebatch == 1) {
					filenameinputdata = remoteoutputfolderichecbatch +
						foldername + "/" + codename + "/" + testcasename + std::to_string(simnum) +
						"_inputdata.dat";
				}
			}
			else {
				if (remotebatch == 0) {
					filenameinputdata = remoteoutputfolderaltint +
						foldername + "/" + codename + "/" + testcasename + std::to_string(simnum) +
						"_inputdata.dat";
				}
				else if (remotebatch == 1) {
					filenameinputdata = remoteoutputfolderaltbatch +
						foldername + "/" + codename + "/" + testcasename + std::to_string(simnum) +
						"_inputdata.dat";
				}
			}
		}
		
		countsearch = simnum;
		
	}else{
		
		if(remoteoutput == 0){
			filenameinputdata = "../../local_output/" +
			foldername + "/" + codename + "/" + testcasename + std::to_string(0) +
			"_inputdata.dat";
		}else if (remoteoutput == 1){
			if (runonichec == 1) {
				if (remotebatch == 0) {
					filenameinputdata = remoteoutputfolderichecint +
						foldername + "/" + codename + "/" + testcasename + std::to_string(0) +
						"_inputdata.dat";
				}
				else if (remotebatch == 1) {
					filenameinputdata = remoteoutputfolderichecbatch +
						foldername + "/" + codename + "/" + testcasename + std::to_string(0) +
						"_inputdata.dat";
				}
			}
			else {
				if (remotebatch == 0) {
					filenameinputdata = remoteoutputfolderaltint +
						foldername + "/" + codename + "/" + testcasename + std::to_string(0) +
						"_inputdata.dat";
				}
				else if (remotebatch == 1) {
					filenameinputdata = remoteoutputfolderaltbatch +
						foldername + "/" + codename + "/" + testcasename + std::to_string(0) +
						"_inputdata.dat";
				}
			}
		}
		
		std::string inputdata = "_inputdata";
		countsearch = fileExists<VALUE_TYPE, INDEX_TYPE>(codename,
		countsearch,inputdata,filenameinputdata,foldername,remotebatch,
		remoteoutput,remoteoutputfolderichecint,remoteoutputfolderichecbatch,
		remoteoutputfolderaltint,remoteoutputfolderaltbatch,runonichec,testcasename);
		
	}

	std::cout<<"\nFilename for input data: "<<filenameinputdata<<std::endl;
	
}

template<typename VALUE_TYPE, typename INDEX_TYPE>
void filenameInputDataChooseFileNumber(
	std::string& filenameinputdata,
	INDEX_TYPE& countsearch,
	INDEX_TYPE& restartsiminput,
	INDEX_TYPE remoteoutput,
	std::string& remoteoutputfolderichecint,
	std::string& remoteoutputfolderichecbatch,
	std::string& remoteoutputfolderaltint,
	std::string& remoteoutputfolderaltbatch,
	INDEX_TYPE& runonichec,
	std::string& foldername,
	std::string& codename,
	std::string& testcasename,
	INDEX_TYPE& simnum,
	INDEX_TYPE remotebatch) {

	if (restartsiminput == 1) {

		if (remoteoutput == 0) {
			filenameinputdata = "../../local_output/" +
				foldername + "/" + codename + "/" + testcasename + std::to_string(simnum) +
				"_inputdata.dat";
		}
		else if (remoteoutput == 1) {
			if (runonichec == 1) {
				if (remotebatch == 0) {
					filenameinputdata = remoteoutputfolderichecint +
						foldername + "/" + codename + "/" + testcasename + std::to_string(simnum) +
						"_inputdata.dat";
				}
				else if (remotebatch == 1) {
					filenameinputdata = remoteoutputfolderichecbatch +
						foldername + "/" + codename + "/" + testcasename + std::to_string(simnum) +
						"_inputdata.dat";
				}
			}
			else {
				if (remotebatch == 0) {
					filenameinputdata = remoteoutputfolderaltint +
						foldername + "/" + codename + "/" + testcasename + std::to_string(simnum) +
						"_inputdata.dat";
				}
				else if (remotebatch == 1) {
					filenameinputdata = remoteoutputfolderaltbatch +
						foldername + "/" + codename + "/" + testcasename + std::to_string(simnum) +
						"_inputdata.dat";
				}
			}
		}

		countsearch = simnum;

	}
	else {

		if (remoteoutput == 0) {
			filenameinputdata = "../../local_output/" +
				foldername + "/" + codename + "/" + testcasename + std::to_string(countsearch) +
				"_inputdata.dat";
		}
		else if (remoteoutput == 1) {
			if (runonichec == 1) {
				if (remotebatch == 0) {
					filenameinputdata = remoteoutputfolderichecint +
						foldername + "/" + codename + "/" + testcasename + std::to_string(countsearch) +
						"_inputdata.dat";
				}
				else if (remotebatch == 1) {
					filenameinputdata = remoteoutputfolderichecbatch +
						foldername + "/" + codename + "/" + testcasename + std::to_string(countsearch) +
						"_inputdata.dat";
				}
			}
			else {
				if (remotebatch == 0) {
					filenameinputdata = remoteoutputfolderaltint +
						foldername + "/" + codename + "/" + testcasename + std::to_string(countsearch) +
						"_inputdata.dat";
				}
				else if (remotebatch == 1) {
					filenameinputdata = remoteoutputfolderaltbatch +
						foldername + "/" + codename + "/" + testcasename + std::to_string(countsearch) +
						"_inputdata.dat";
				}
			}
		}

	}

	std::cout << "\nFilename for input data: " << filenameinputdata << std::endl;

}

template<typename VALUE_TYPE, typename INDEX_TYPE>
void filename3D(
	INDEX_TYPE remoteoutput,
	std::string &remoteoutputfolderichecint,
	std::string &remoteoutputfolderichecbatch,
	std::string &remoteoutputfolderaltint,
	std::string &remoteoutputfolderaltbatch,
	INDEX_TYPE &runonichec,
	std::string &filename3d,
	std::string &foldername,
	std::string &codename,
	std::string &testcasename,
	INDEX_TYPE &countsearch,
	std::string &filenameend,
	INDEX_TYPE remotebatch,
	std::string &filenameinfo){
	
	if(remoteoutput == 0){
		filename3d = "../../local_output/" +
		foldername + "/" + codename + "/" + testcasename +
		std::to_string(countsearch) + filenameend;
	}else if(remoteoutput == 1){
		if (runonichec == 1) {
			if (remotebatch == 0) {
				filename3d = remoteoutputfolderichecint +
					foldername + "/" + codename + "/" + testcasename +
					std::to_string(countsearch) + filenameend;
			}
			else if (remotebatch == 1) {
				filename3d = remoteoutputfolderichecbatch +
					foldername + "/" + codename + "/" + testcasename +
					std::to_string(countsearch) + filenameend;
			}
		}else {
			if (remotebatch == 0) {
				filename3d = remoteoutputfolderaltint +
					foldername + "/" + codename + "/" + testcasename +
					std::to_string(countsearch) + filenameend;
			}
			else if (remotebatch == 1) {
				filename3d = remoteoutputfolderaltbatch +
					foldername + "/" + codename + "/" + testcasename +
					std::to_string(countsearch) + filenameend;
			}
		}
	}
	std::cout<<filenameinfo<<filename3d<<std::endl;
	
}

template<typename VALUE_TYPE, typename INDEX_TYPE>
void filename3DRestartInput(
	INDEX_TYPE remoteoutput,
	std::string &remoteoutputfolderichecint,
	std::string &remoteoutputfolderichecbatch,
	std::string &remoteoutputfolderaltint,
	std::string &remoteoutputfolderaltbatch,
	INDEX_TYPE &runonichec,
	std::string &filename3d,
	std::string &foldername,
	std::string &codename,
	std::string &testcasename,
	INDEX_TYPE &countsearch,
	std::string &filenameend,
	INDEX_TYPE countsearchrestart,
	INDEX_TYPE remotebatch,
	std::string &filenameinfo){
	
	if(remoteoutput == 0){
		filename3d = "../../local_output/" +
		foldername + "/" + codename + "/" + testcasename +
		std::to_string(countsearch) + filenameend +
		std::to_string(countsearchrestart);
	}else if(remoteoutput == 1){
		if (runonichec == 1) {
			if (remotebatch == 0) {
				filename3d = remoteoutputfolderichecint +
					foldername + "/" + codename + "/" + testcasename +
					std::to_string(countsearch) + filenameend +
					std::to_string(countsearchrestart);
			}
			else if (remotebatch == 1) {
				filename3d = remoteoutputfolderichecbatch +
					foldername + "/" + codename + "/" + testcasename +
					std::to_string(countsearch) + filenameend +
					std::to_string(countsearchrestart);
			}
		}else {
			if (remotebatch == 0) {
				filename3d = remoteoutputfolderaltint +
					foldername + "/" + codename + "/" + testcasename +
					std::to_string(countsearch) + filenameend +
					std::to_string(countsearchrestart);
			}
			else if (remotebatch == 1) {
				filename3d = remoteoutputfolderaltbatch +
					foldername + "/" + codename + "/" + testcasename +
					std::to_string(countsearch) + filenameend +
					std::to_string(countsearchrestart);
			}
		}
	}
	std::cout<<filenameinfo<<filename3d<<std::endl;
	
}

template<typename VALUE_TYPE, typename INDEX_TYPE>
void filename3DRestartInputVector(
	INDEX_TYPE remoteoutput,
	std::string &remoteoutputfolderichecint,
	std::string &remoteoutputfolderichecbatch,
	std::string &remoteoutputfolderaltint,
	std::string &remoteoutputfolderaltbatch,
	INDEX_TYPE &runonichec,
	std::vector<std::string> &filename3d,
	std::string &foldername,
	std::string &codename,
	std::string &testcasename,
	INDEX_TYPE &countsearch,
	std::string &filenameend,
	INDEX_TYPE countsearchrestart,
	INDEX_TYPE remotebatch,
	std::string &filenameinfo,
	INDEX_TYPE &p){
	
	if(remoteoutput == 0){
		filename3d.push_back("../../local_output/" +
		foldername + "/" + codename + "/" + testcasename +
		std::to_string(countsearch) + filenameend +
		std::to_string(countsearchrestart));
	}else if(remoteoutput == 1){
		if (runonichec == 1) {
			if (remotebatch == 0) {
				filename3d.push_back(remoteoutputfolderichecint +
					foldername + "/" + codename + "/" + testcasename +
					std::to_string(countsearch) + filenameend +
					std::to_string(countsearchrestart));
			}
			else if (remotebatch == 1) {
				filename3d.push_back(remoteoutputfolderichecbatch +
					foldername + "/" + codename + "/" + testcasename +
					std::to_string(countsearch) + filenameend +
					std::to_string(countsearchrestart));
			}
		}else {
			if (remotebatch == 0) {
				filename3d.push_back(remoteoutputfolderaltint +
					foldername + "/" + codename + "/" + testcasename +
					std::to_string(countsearch) + filenameend +
					std::to_string(countsearchrestart));
			}
			else if (remotebatch == 1) {
				filename3d.push_back(remoteoutputfolderaltbatch +
					foldername + "/" + codename + "/" + testcasename +
					std::to_string(countsearch) + filenameend +
					std::to_string(countsearchrestart));
			}
		}
	}
	std::cout<<filenameinfo<<filename3d[p]<<std::endl;
	
}

void filename3DRestartOutputSearch(
	INDEX_TYPE remoteoutput,
	std::string &remoteoutputfolderichecint,
	std::string &remoteoutputfolderichecbatch,
	std::string &remoteoutputfolderaltint,
	std::string &remoteoutputfolderaltbatch,
	INDEX_TYPE &runonichec,
	std::string &filename3d,
	std::string &foldername,
	std::string &codename,
	std::string &testcasename,
	INDEX_TYPE &countsearch,
	std::string &filenameend,
	INDEX_TYPE &restartoutputnum,
	INDEX_TYPE remotebatch,
	INDEX_TYPE &countsearchrestart,
	std::string &filenameinfo){
	
	if(remoteoutput == 0){
		filename3d = "../../local_output/" +
		foldername + "/" + codename + "/" + testcasename +
		std::to_string(countsearch) + filenameend +
		std::to_string(restartoutputnum);
	}else if(remoteoutput == 1){
		if (runonichec == 1) {
			if (remotebatch == 0) {
				filename3d = remoteoutputfolderichecint +
					foldername + "/" + codename + "/" + testcasename +
					std::to_string(countsearch) + filenameend +
					std::to_string(restartoutputnum);
			}
			else if (remotebatch == 1) {
				filename3d = remoteoutputfolderichecbatch +
					foldername + "/" + codename + "/" + testcasename +
					std::to_string(countsearch) + filenameend +
					std::to_string(restartoutputnum);
			}
		}else {
			if (remotebatch == 0) {
				filename3d = remoteoutputfolderaltint +
					foldername + "/" + codename + "/" + testcasename +
					std::to_string(countsearch) + filenameend +
					std::to_string(restartoutputnum);
			}
			else if (remotebatch == 1) {
				filename3d = remoteoutputfolderaltbatch +
					foldername + "/" + codename + "/" + testcasename +
					std::to_string(countsearch) + filenameend +
					std::to_string(restartoutputnum);
			}
		}
	}
	
	countsearchrestart = restartoutputnum;
	std::string restartdata = filenameend;
	countsearchrestart = restartFileExists<VALUE_TYPE, INDEX_TYPE>(
	codename,countsearch,countsearchrestart,restartdata,
	filename3d,foldername,remotebatch,
	remoteoutput,remoteoutputfolderichecint,
	remoteoutputfolderichecbatch,remoteoutputfolderaltint,
	remoteoutputfolderaltbatch,runonichec,testcasename);
	
	std::cout<<filenameinfo<<filename3d<<std::endl;
	
}

template<typename VALUE_TYPE, typename INDEX_TYPE>
void filename3DRestartOutput(
	INDEX_TYPE remoteoutput,
	std::string &remoteoutputfolderichecint,
	std::string &remoteoutputfolderichecbatch,
	std::string &remoteoutputfolderaltint,
	std::string &remoteoutputfolderaltbatch,
	INDEX_TYPE &runonichec,
	std::string &filename3d,
	std::string &foldername,
	std::string &codename,
	std::string &testcasename,
	INDEX_TYPE &countsearch,
	std::string &filenameend,
	INDEX_TYPE countsearchrestart,
	INDEX_TYPE remotebatch,
	std::string &filenameinfo){
	
	if(remoteoutput == 0){
		filename3d = "../../local_output/" +
		foldername + "/" + codename + "/" + testcasename +
		std::to_string(countsearch) + filenameend +
		std::to_string(countsearchrestart);
	}else if(remoteoutput == 1){
		if (runonichec == 1) {
			if (remotebatch == 0) {
				filename3d = remoteoutputfolderichecint +
					foldername + "/" + codename + "/" + testcasename +
					std::to_string(countsearch) + filenameend +
					std::to_string(countsearchrestart);
			}
			else if (remotebatch == 1) {
				filename3d = remoteoutputfolderichecbatch +
					foldername + "/" + codename + "/" + testcasename +
					std::to_string(countsearch) + filenameend +
					std::to_string(countsearchrestart);
			}
		}else {
			if (remotebatch == 0) {
				filename3d = remoteoutputfolderaltint +
					foldername + "/" + codename + "/" + testcasename +
					std::to_string(countsearch) + filenameend +
					std::to_string(countsearchrestart);
			}
			else if (remotebatch == 1) {
				filename3d = remoteoutputfolderaltbatch +
					foldername + "/" + codename + "/" + testcasename +
					std::to_string(countsearch) + filenameend +
					std::to_string(countsearchrestart);
			}
		}
	}
	std::cout<<filenameinfo<<filename3d<<std::endl;
	
}

template<typename VALUE_TYPE, typename INDEX_TYPE>
void filename3DRestartOutputVector(
	INDEX_TYPE remoteoutput,
	std::string &remoteoutputfolderichecint,
	std::string &remoteoutputfolderichecbatch,
	std::string &remoteoutputfolderaltint,
	std::string &remoteoutputfolderaltbatch,
	INDEX_TYPE &runonichec,
	std::vector<std::string> &filename3d,
	std::string &foldername,
	std::string &codename,
	std::string &testcasename,
	INDEX_TYPE &countsearch,
	std::string &filenameend,
	INDEX_TYPE countsearchrestart,
	INDEX_TYPE remotebatch,
	std::string &filenameinfo,
	INDEX_TYPE &p){
	
	if(remoteoutput == 0){
		filename3d.push_back("../../local_output/" +
		foldername + "/" + codename + "/" + testcasename +
		std::to_string(countsearch) + filenameend +
		std::to_string(countsearchrestart));
	}else if(remoteoutput == 1){
		if (runonichec == 1) {
			if (remotebatch == 0) {
				filename3d.push_back(remoteoutputfolderichecint +
					foldername + "/" + codename + "/" + testcasename +
					std::to_string(countsearch) + filenameend +
					std::to_string(countsearchrestart));
			}
			else if (remotebatch == 1) {
				filename3d.push_back(remoteoutputfolderichecbatch +
					foldername + "/" + codename + "/" + testcasename +
					std::to_string(countsearch) + filenameend +
					std::to_string(countsearchrestart));
			}
		}else {
			if (remotebatch == 0) {
				filename3d.push_back(remoteoutputfolderaltint +
					foldername + "/" + codename + "/" + testcasename +
					std::to_string(countsearch) + filenameend +
					std::to_string(countsearchrestart));
			}
			else if (remotebatch == 1) {
				filename3d.push_back(remoteoutputfolderaltbatch +
					foldername + "/" + codename + "/" + testcasename +
					std::to_string(countsearch) + filenameend +
					std::to_string(countsearchrestart));
			}
		}
	}
	std::cout<<filenameinfo<<filename3d[p]<<std::endl;
	
}
