#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <iomanip>
#include <sys/time.h>

#include "../3dsetup/typedefs.h"

/// Checks if file containing countsearch number exists. If already exists,
/// increments countsearch and calls function recursively to check again until
/// file doesn't exit

template<typename VALUE_TYPE, typename INDEX_TYPE>
INDEX_TYPE fileExists(
	std::string &codename,
	INDEX_TYPE &countsearch,
	std::string &datatype,
	std::string &fname,
	std::string &foldername,
	INDEX_TYPE &remotebatch,
	INDEX_TYPE &remoteoutput,
	std::string &remoteoutputfolderichecint,
	std::string &remoteoutputfolderichecbatch,
	std::string &remoteoutputfolderaltint,
	std::string &remoteoutputfolderaltbatch,
	INDEX_TYPE &runonichec,
	std::string &testcasename){
	
	std::ifstream tester;
	tester.open(fname);
	
	if(tester.is_open()){
		
		tester.close();
		countsearch = countsearch + 1;
		
		if(remoteoutput == 0){
			
			fname = std::string("../../") +
					"local_output/" +
					foldername +
					"/" +
					codename +
					"/" +
					testcasename +
					std::to_string(countsearch) +
					datatype +
					".dat";
					
		}else if (remoteoutput == 1){

			if (runonichec == 1) {

				if (remotebatch == 0) {

					fname = remoteoutputfolderichecint +
						foldername +
						"/" +
						codename +
						"/" +
						testcasename +
						std::to_string(countsearch) +
						datatype +
						".dat";

				}
				else if (remotebatch == 1) {

					fname = remoteoutputfolderichecbatch +
						foldername +
						"/" +
						codename +
						"/" +
						testcasename +
						std::to_string(countsearch) +
						datatype +
						".dat";

				}

			}else {

				if (remotebatch == 0) {

					fname = remoteoutputfolderaltint +
						foldername +
						"/" +
						codename +
						"/" +
						testcasename +
						std::to_string(countsearch) +
						datatype +
						".dat";

				}
				else if (remotebatch == 1) {

					fname = remoteoutputfolderaltbatch +
						foldername +
						"/" +
						codename +
						"/" +
						testcasename +
						std::to_string(countsearch) +
						datatype +
						".dat";

				}

			}
		}
		
		fileExists<VALUE_TYPE, INDEX_TYPE>(codename,countsearch,datatype,fname,
		foldername,remotebatch,remoteoutput,remoteoutputfolderichecint,
		remoteoutputfolderichecbatch,remoteoutputfolderaltint,
		remoteoutputfolderaltbatch,runonichec,testcasename);
		
	}
	
	return countsearch;
	
}

/// Checks if RESTART file containing countsearch number exists. If already exists,
/// increments countsearch and calls function recursively to check again until
/// file doesn't exit

template<typename VALUE_TYPE, typename INDEX_TYPE>
INDEX_TYPE restartFileExists(
	std::string &codename,
	INDEX_TYPE &countsearch,
	INDEX_TYPE &countsearchrestart,
	std::string &datatype,
	std::string &fname,
	std::string &foldername,
	INDEX_TYPE &remotebatch,
	INDEX_TYPE &remoteoutput,
	std::string &remoteoutputfolderichecint,
	std::string &remoteoutputfolderichecbatch,
	std::string &remoteoutputfolderaltint,
	std::string &remoteoutputfolderaltbatch,
	INDEX_TYPE &runonichec,
	std::string &testcasename){
	
	std::ifstream tester;
	tester.open(fname);
	
	if(tester.is_open()){
		
		tester.close();
		countsearchrestart = countsearchrestart + 1;
		
		if(remoteoutput == 0){
			
			fname = std::string("../../") +
					"local_output/" +
					foldername +
					"/" +
					codename +
					"/" +
					testcasename +
					std::to_string(countsearch) +
					datatype +
					".RESTART" +
					std::to_string(countsearchrestart);
					
		}else if (remoteoutput == 1){

			if (runonichec == 1) {

				if (remotebatch == 0) {

					fname = remoteoutputfolderichecint +
						foldername +
						"/" +
						codename +
						"/" +
						testcasename +
						std::to_string(countsearch) +
						datatype +
						".RESTART" +
						std::to_string(countsearchrestart);

				}
				else if (remotebatch == 1) {

					fname = remoteoutputfolderichecbatch +
						foldername +
						"/" +
						codename +
						"/" +
						testcasename +
						std::to_string(countsearch) +
						datatype +
						".RESTART" +
						std::to_string(countsearchrestart);

				}

			}else {

				if (remotebatch == 0) {

					fname = remoteoutputfolderaltint +
						foldername +
						"/" +
						codename +
						"/" +
						testcasename +
						std::to_string(countsearch) +
						datatype +
						".RESTART" +
						std::to_string(countsearchrestart);

				}
				else if (remotebatch == 1) {

					fname = remoteoutputfolderaltbatch +
						foldername +
						"/" +
						codename +
						"/" +
						testcasename +
						std::to_string(countsearch) +
						datatype +
						".RESTART" +
						std::to_string(countsearchrestart);

				}

			}
		}
		
		restartFileExists<VALUE_TYPE, INDEX_TYPE>(codename,countsearch,countsearchrestart,datatype,fname,
		foldername,remotebatch,remoteoutput,remoteoutputfolderichecint,remoteoutputfolderichecbatch,
		remoteoutputfolderaltint,remoteoutputfolderaltbatch,runonichec, testcasename);
		
	}
	
	return countsearchrestart;
	
}
