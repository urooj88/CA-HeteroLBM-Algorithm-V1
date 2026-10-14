#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <iomanip>
#include <sys/time.h>
#include <string>
#include <sstream>

#include "../3dsetup/typedefs.h"

#include <algorithm>
#include <cctype>
#include <stdexcept>

// ============================================================
// Small string helpers
// ============================================================
inline std::string trim(const std::string& s) {
    auto wsfront = std::find_if_not(s.begin(), s.end(),
        [](int c) { return std::isspace(c); });
    auto wsback = std::find_if_not(s.rbegin(), s.rend(),
        [](int c) { return std::isspace(c); }).base();
    return (wsback <= wsfront ? std::string() : std::string(wsfront, wsback));
}

inline std::vector<std::string> tokenize(const std::string& expr) {
    std::vector<std::string> tokens;
    std::string temp;
    bool expectNumber = true;

    for (size_t i = 0; i < expr.length(); ++i) {
        char ch = expr[i];
        if (isspace(ch)) continue;

        if (ch == 'M' && i + 3 < expr.length() && expr.substr(i, 4) == "M_PI") {
            if (!temp.empty() && temp == "-") {
                tokens.push_back("-" + std::to_string(M_PI));
                temp.clear();
            } else {
                tokens.push_back(std::to_string(M_PI));
            }
            i += 3;
            expectNumber = false;
        } else if (isdigit(ch) || ch == '.' || (expectNumber && ch == '-')) {
            temp += ch;
            expectNumber = false;
        } else {
            if (!temp.empty()) {
                tokens.push_back(temp);
                temp.clear();
            }
            tokens.push_back(std::string(1, ch));
            expectNumber = true;
        }
    }

    if (!temp.empty()) {
        tokens.push_back(temp);
    }

    return tokens;
}

inline double evaluate(const std::vector<std::string>& tokens) {
    double result = 0.0;
    std::string lastOp = "+";

    for (const auto& token : tokens) {
        if (token == "+" || token == "-" || token == "*" || token == "/") {
            lastOp = token;
        } else {
            double value = 0.0;
            if (token == "M_PI") {
                value = M_PI;
            } else {
                value = std::stod(token);
            }

            if (lastOp == "+") result += value;
            else if (lastOp == "-") result -= value;
            else if (lastOp == "*") result *= value;
            else if (lastOp == "/") result /= value;
        }
    }

    return result;
}

// ============================================================
// Heterogeneous runtime configuration
// ============================================================
template<typename VALUE_TYPE, typename INDEX_TYPE>
struct HeteroRuntimeConfig {
    INDEX_TYPE heterogeneous_mode = 0;   // 0 off, 1 on
    VALUE_TYPE gpu0_weight = VALUE_TYPE(0.45);
    VALUE_TYPE gpu1_weight = VALUE_TYPE(0.45);
    VALUE_TYPE cpu_weight  = VALUE_TYPE(0.10);
    INDEX_TYPE gpu0_id = 0;
    INDEX_TYPE gpu1_id = 1;
    INDEX_TYPE cpu_threads = 16;
};

template<typename VALUE_TYPE, typename INDEX_TYPE>
void readHeteroRuntimeConfig(
    std::string &inputname,
    HeteroRuntimeConfig<VALUE_TYPE, INDEX_TYPE> &cfg)
{
    std::ifstream input(inputname);
    if (!input.is_open()) {
        std::cout << "Unable to open input.txt for heterogeneous runtime config" << std::endl;
        return;
    }

    std::string line;
    while (std::getline(input, line)) {
        line = trim(line);
        if (line.empty()) continue;

        std::istringstream iss(line);
        std::string paramname;
        VALUE_TYPE value;

        if (!(iss >> paramname >> value)) continue;

        if (paramname == "Heterogeneous_mode:") {
            cfg.heterogeneous_mode = static_cast<INDEX_TYPE>(value);
        } else if (paramname == "GPU0_weight:") {
            cfg.gpu0_weight = static_cast<VALUE_TYPE>(value);
        } else if (paramname == "GPU1_weight:") {
            cfg.gpu1_weight = static_cast<VALUE_TYPE>(value);
        } else if (paramname == "CPU_weight:") {
            cfg.cpu_weight = static_cast<VALUE_TYPE>(value);
        } else if (paramname == "GPU0_id:") {
            cfg.gpu0_id = static_cast<INDEX_TYPE>(value);
        } else if (paramname == "GPU1_id:") {
            cfg.gpu1_id = static_cast<INDEX_TYPE>(value);
        } else if (paramname == "CPU_threads:") {
            cfg.cpu_threads = static_cast<INDEX_TYPE>(value);
        }
    }

    input.close();
}

// ============================================================
// Reads in data from specified input file
// ============================================================
template<typename VALUE_TYPE, typename INDEX_TYPE>
void readInData(
    VALUE_TYPE &convfactrhocrho,
    INDEX_TYPE &delxdim,
    INDEX_TYPE &ibicosubdivisions,
    INDEX_TYPE &ibnumobjects,
    std::vector<VALUE_TYPE> &ibphysobjectloc,
    std::string &inputname,
    INDEX_TYPE &multitimestepoutput,
    INDEX_TYPE &Ncx,
    INDEX_TYPE &Ncy,
    INDEX_TYPE &Ncz,
    INDEX_TYPE &objectplaced,
    INDEX_TYPE &periodicinx,
    INDEX_TYPE &periodiciny,
    INDEX_TYPE &periodicinz,
    VALUE_TYPE &physchardimtouse,
    VALUE_TYPE &physcharvel,
    VALUE_TYPE &physlengthx,
    VALUE_TYPE &physlengthy,
    VALUE_TYPE &physlengthz,
    VALUE_TYPE &physobjectcritdim,
    VALUE_TYPE &physobjectlocx,
    VALUE_TYPE &physobjectlocy,
    VALUE_TYPE &physobjectlocz,
    VALUE_TYPE &physrho,
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
    VALUE_TYPE &physvisc,
    INDEX_TYPE &relaxtype,
    INDEX_TYPE &remoteoutput,
    INDEX_TYPE &simtype,
    INDEX_TYPE &stype,
    VALUE_TYPE &tau)
{
    std::string readphyslengthx ("Physical_length_in_x_direction:");
    std::string readphyslengthy ("Physical_length_in_y_direction:");
    std::string readphyslengthz ("Physical_length_in_z_direction:");

    std::string readcellsx ("Number_of_cells_in_x_direction:");
    std::string readcellsy ("Number_of_cells_in_y_direction:");
    std::string readcellsz ("Number_of_cells_in_z_direction:");

    std::string readperiodicinx ("Periodic_in_x:");
    std::string readperiodiciny ("Periodic_in_y:");
    std::string readperiodicinz ("Periodic_in_z:");

    std::string readphysrho ("Physical_density:");
    std::string readconvfactrhocrho ("Density_conversion_factor:");
    std::string readphysvisc ("Physical_viscosity:");

    std::string readphysvxback ("Physical_vx_back:");
    std::string readphysvxbottom ("Physical_vx_bottom:");
    std::string readphysvxfront ("Physical_vx_front:");
    std::string readphysvxleft ("Physical_vx_Left:");
    std::string readphysvxleft2 ("Physical_vx_left:");
    std::string readphysvxright ("Physical_vx_right:");
    std::string readphysvxtop ("Physical_vx_top:");

    std::string readphysvyback ("Physical_vy_back:");
    std::string readphysvybottom ("Physical_vy_bottom:");
    std::string readphysvyfront ("Physical_vy_front:");
    std::string readphysvyleft ("Physical_vy_left:");
    std::string readphysvyright ("Physical_vy_right:");
    std::string readphysvytop ("Physical_vy_top:");

    std::string readphysvzback ("Physical_vz_back:");
    std::string readphysvzbottom ("Physical_vz_bottom:");
    std::string readphysvzfront ("Physical_vz_front:");
    std::string readphysvzleft ("Physical_vz_left:");
    std::string readphysvzright ("Physical_vz_right:");
    std::string readphysvztop ("Physical_vz_top:");

    std::string readtau ("Relaxation_factor:");
    std::string readrelaxtype ("Relaxation_type:");
    std::string readstype ("S_type:");
    std::string readsimtype ("Simulation_type:");
    std::string readdelxdim ("Del_x_dimension:");
    std::string readphyschardimtouse ("Physical_characteristic_dimension_to_use:");
    std::string readphyscharvel ("Physical_characteristic_velocity_to_use:");
    std::string readmultitimestepoutput ("Multiple_timestep_output:");
    std::string readremoteoutput ("Remote_output:");

    std::string readobjectplaced ("Object_placed:");
    std::string readphysobjectcritdim ("Physical_object_critical_dimension:");
    std::string readphysobjectlocx ("Physical_object_location_x:");
    std::string readphysobjectlocy ("Physical_object_location_y:");
    std::string readphysobjectlocz ("Physical_object_location_z:");

    std::string readibnumobjects ("IB_number_of_objects:");
    std::string readibicosubdivisions ("IB_icosphere_refinement:");
    std::string readibphysobjectlocx;
    std::string readibphysobjectlocy;
    std::string readibphysobjectlocz;

    std::string paramname;
    VALUE_TYPE value;

    std::ifstream input;
    input.open(inputname);

    if (input.is_open()) {
        while (!input.eof()) {
            input >> paramname >> value;
            if (!input.good()) break;

            if (paramname.compare(readphyslengthx) == 0) {
                physlengthx = value;
            } else if (paramname.compare(readphyslengthy) == 0) {
                physlengthy = value;
            } else if (paramname.compare(readphyslengthz) == 0) {
                physlengthz = value;

            } else if (paramname.compare(readcellsx) == 0) {
                Ncx = static_cast<INDEX_TYPE>(value);
            } else if (paramname.compare(readcellsy) == 0) {
                Ncy = static_cast<INDEX_TYPE>(value);
            } else if (paramname.compare(readcellsz) == 0) {
                Ncz = static_cast<INDEX_TYPE>(value);

            } else if (paramname.compare(readperiodicinx) == 0) {
                periodicinx = static_cast<INDEX_TYPE>(value);
            } else if (paramname.compare(readperiodiciny) == 0) {
                periodiciny = static_cast<INDEX_TYPE>(value);
            } else if (paramname.compare(readperiodicinz) == 0) {
                periodicinz = static_cast<INDEX_TYPE>(value);

            } else if (paramname.compare(readphysrho) == 0) {
                physrho = value;
            } else if (paramname.compare(readphysvisc) == 0) {
                physvisc = value;
            } else if (paramname.compare(readconvfactrhocrho) == 0) {
                convfactrhocrho = value;

            } else if (paramname.compare(readphysvxback) == 0) {
                physvxback = value;
            } else if (paramname.compare(readphysvxbottom) == 0) {
                physvxbottom = value;
            } else if (paramname.compare(readphysvxfront) == 0) {
                physvxfront = value;
            } else if (paramname.compare(readphysvxleft) == 0 || paramname.compare(readphysvxleft2) == 0) {
                physvxleft = value;
            } else if (paramname.compare(readphysvxright) == 0) {
                physvxright = value;
            } else if (paramname.compare(readphysvxtop) == 0) {
                physvxtop = value;

            } else if (paramname.compare(readphysvyback) == 0) {
                physvyback = value;
            } else if (paramname.compare(readphysvybottom) == 0) {
                physvybottom = value;
            } else if (paramname.compare(readphysvyfront) == 0) {
                physvyfront = value;
            } else if (paramname.compare(readphysvyleft) == 0) {
                physvyleft = value;
            } else if (paramname.compare(readphysvyright) == 0) {
                physvyright = value;
            } else if (paramname.compare(readphysvytop) == 0) {
                physvytop = value;

            } else if (paramname.compare(readphysvzback) == 0) {
                physvzback = value;
            } else if (paramname.compare(readphysvzbottom) == 0) {
                physvzbottom = value;
            } else if (paramname.compare(readphysvzfront) == 0) {
                physvzfront = value;
            } else if (paramname.compare(readphysvzleft) == 0) {
                physvzleft = value;
            } else if (paramname.compare(readphysvzright) == 0) {
                physvzright = value;
            } else if (paramname.compare(readphysvztop) == 0) {
                physvztop = value;

            } else if (paramname.compare(readtau) == 0) {
                tau = value;
            } else if (paramname.compare(readrelaxtype) == 0) {
                relaxtype = static_cast<INDEX_TYPE>(value);
            } else if (paramname.compare(readstype) == 0) {
                stype = static_cast<INDEX_TYPE>(value);
            } else if (paramname.compare(readsimtype) == 0) {
                simtype = static_cast<INDEX_TYPE>(value);
            } else if (paramname.compare(readdelxdim) == 0) {
                delxdim = static_cast<INDEX_TYPE>(value);
            } else if (paramname.compare(readphyschardimtouse) == 0) {
                physchardimtouse = value;
            } else if (paramname.compare(readphyscharvel) == 0) {
                physcharvel = value;
            } else if (paramname.compare(readmultitimestepoutput) == 0) {
                multitimestepoutput = static_cast<INDEX_TYPE>(value);
            } else if (paramname.compare(readremoteoutput) == 0) {
                remoteoutput = static_cast<INDEX_TYPE>(value);

            } else if (paramname.compare(readobjectplaced) == 0) {
                objectplaced = static_cast<INDEX_TYPE>(value);
            } else if (paramname.compare(readphysobjectcritdim) == 0) {
                physobjectcritdim = value;
            } else if (paramname.compare(readphysobjectlocx) == 0) {
                physobjectlocx = value;
            } else if (paramname.compare(readphysobjectlocy) == 0) {
                physobjectlocy = value;
            } else if (paramname.compare(readphysobjectlocz) == 0) {
                physobjectlocz = value;

            } else if (paramname.compare(readibicosubdivisions) == 0) {
                ibicosubdivisions = static_cast<INDEX_TYPE>(value);
            } else if (paramname.compare(readibnumobjects) == 0) {
                ibnumobjects = static_cast<INDEX_TYPE>(value);
                for (INDEX_TYPE i = 0; i < ibnumobjects; i++) {
                    readibphysobjectlocx = "IB_physical_object_location_x_";
                    readibphysobjectlocy = "IB_physical_object_location_y_";
                    readibphysobjectlocz = "IB_physical_object_location_z_";
                    readibphysobjectlocx += std::to_string(i) + ":";
                    readibphysobjectlocy += std::to_string(i) + ":";
                    readibphysobjectlocz += std::to_string(i) + ":";
                }
            } else if (paramname.compare(readibphysobjectlocx) == 0) {
                ibphysobjectloc.push_back(value);
            } else if (paramname.compare(readibphysobjectlocy) == 0) {
                ibphysobjectloc.push_back(value);
            } else if (paramname.compare(readibphysobjectlocz) == 0) {
                ibphysobjectloc.push_back(value);
            }
        }

        input.close();

    } else {
        std::cout << "Unable to open input.txt" << std::endl;
    }
}

// ============================================================
// Remote batch flag
// ============================================================
template<typename VALUE_TYPE, typename INDEX_TYPE>
void readInRemoteBatch(
    std::string &remotebatchname,
    INDEX_TYPE &remotebatch)
{
    std::string readremotebatch("Remote_batch:");

    std::string paramname;
    VALUE_TYPE value;

    std::ifstream input;
    input.open(remotebatchname);

    if (input.is_open()) {
        while (!input.eof()) {
            input >> paramname >> value;
            if (!input.good()) break;

            if (paramname.compare(readremotebatch) == 0) {
                remotebatch = static_cast<INDEX_TYPE>(value);
            }
        }

        input.close();

    } else {
        std::cout << "Unable to open remote_batch.txt - executing program in location other than home/" << std::endl;
    }
}

// ============================================================
// Stop/restart controls
// ============================================================
template<typename VALUE_TYPE, typename INDEX_TYPE>
void readInStopRestartSim(
    std::string &stoprestartsimulationname,
    INDEX_TYPE &stopsim,
    INDEX_TYPE &simnum,
    INDEX_TYPE &restartsiminput,
    INDEX_TYPE &restartsiminputnum,
    INDEX_TYPE &restartsimoutput)
{
    std::string readstopsim ("Stop_simulation:");
    std::string readrestartsiminput ("Restart_simulation_input:");
    std::string readrestartsiminputnum ("Restart_simulation_input_number:");
    std::string readrestartsimoutput ("Restart_simulation_output:");
    std::string readsimnum ("Simulation_number_based_on_Tecplot_output:");

    std::string paramname;
    VALUE_TYPE value;

    std::ifstream input;
    input.open(stoprestartsimulationname);

    if (input.is_open()) {
        while (!input.eof()) {
            input >> paramname >> value;
            if (!input.good()) break;

            if (paramname.compare(readstopsim) == 0) {
                stopsim = static_cast<INDEX_TYPE>(value);
            } else if (paramname.compare(readrestartsiminput) == 0) {
                restartsiminput = static_cast<INDEX_TYPE>(value);
            } else if (paramname.compare(readrestartsiminputnum) == 0) {
                restartsiminputnum = static_cast<INDEX_TYPE>(value);
            } else if (paramname.compare(readrestartsimoutput) == 0) {
                restartsimoutput = static_cast<INDEX_TYPE>(value);
            } else if (paramname.compare(readsimnum) == 0) {
                simnum = static_cast<INDEX_TYPE>(value);
            }
        }

        input.close();

    } else {
        std::cout << "Unable to open simulation STOPRESTART file!" << std::endl;
    }
}

// ============================================================
// Restart input/output helpers
// These are kept from your original file
// ============================================================
template<typename VALUE_TYPE, typename INDEX_TYPE>
void restartInput3DLBDensities(
    std::string &filename3dlbdensitiesrestartinput,
    std::ifstream &f3dlbrhori,
    INDEX_TYPE &Ncx,
    INDEX_TYPE &Ncy,
    INDEX_TYPE &Ncz,
    INDEX_TYPE &Nnx,
    INDEX_TYPE &Nny,
    INDEX_TYPE &Nnz,
    INDEX_TYPE &periodicinx,
    INDEX_TYPE &periodiciny,
    INDEX_TYPE &periodicinz,
    INDEX_TYPE &step,
    VALUE_TYPE *rho)
{
    std::string fname;
    fname = filename3dlbdensitiesrestartinput;
    f3dlbrhori.open(fname.c_str());

    INDEX_TYPE ixyz;

    INDEX_TYPE startnodex, endnodex;
    if (periodicinx == 0) { startnodex = 0; endnodex = Nnx - 1; }
    else                  { startnodex = 1; endnodex = Nnx - 2; }

    INDEX_TYPE startnodey, endnodey;
    if (periodiciny == 0) { startnodey = 0; endnodey = Nny - 1; }
    else                  { startnodey = 1; endnodey = Nny - 2; }

    INDEX_TYPE startnodez, endnodez;
    if (periodicinz == 0) { startnodez = 0; endnodez = Nnz - 1; }
    else                  { startnodez = 1; endnodez = Nnz - 2; }

    if (f3dlbrhori.is_open()) { f3dlbrhori.close(); }
    else std::cout << "Unable to open density RESTART input file!" << std::endl;

    fname = filename3dlbdensitiesrestartinput;
    f3dlbrhori.open(fname.c_str());

    f3dlbrhori >> step;
    f3dlbrhori.ignore();
    std::string variable;
    f3dlbrhori >> variable;

    for (INDEX_TYPE k = startnodez; k <= endnodez; k++) {
        for (INDEX_TYPE j = startnodey; j <= endnodey; j++) {
            for (INDEX_TYPE i = startnodex; i <= endnodex; i++) {
                ixyz = i + j * Nnx + k * Nnx * Nny;
                f3dlbrhori >> rho[ixyz];
            }
        }
    }

    f3dlbrhori.close();
}

template<typename VALUE_TYPE, typename INDEX_TYPE>
void restartInput3DLBErrorVelocities(
    std::string &filename3dlberrorvelocitiesrestartinput,
    std::ifstream &f3dlbevri,
    INDEX_TYPE &Ncx,
    INDEX_TYPE &Ncy,
    INDEX_TYPE &Ncz,
    INDEX_TYPE &Nnx,
    INDEX_TYPE &Nny,
    INDEX_TYPE &Nnz,
    INDEX_TYPE &periodicinx,
    INDEX_TYPE &periodiciny,
    INDEX_TYPE &periodicinz,
    INDEX_TYPE &step,
    VALUE_TYPE *vxerr,
    VALUE_TYPE *vyerr,
    VALUE_TYPE *vzerr)
{
    std::string fname;
    fname = filename3dlberrorvelocitiesrestartinput;
    f3dlbevri.open(fname.c_str());

    INDEX_TYPE ixyz;

    INDEX_TYPE startnodex, endnodex;
    if (periodicinx == 0) { startnodex = 0; endnodex = Nnx - 1; }
    else                  { startnodex = 1; endnodex = Nnx - 2; }

    INDEX_TYPE startnodey, endnodey;
    if (periodiciny == 0) { startnodey = 0; endnodey = Nny - 1; }
    else                  { startnodey = 1; endnodey = Nny - 2; }

    INDEX_TYPE startnodez, endnodez;
    if (periodicinz == 0) { startnodez = 0; endnodez = Nnz - 1; }
    else                  { startnodez = 1; endnodez = Nnz - 2; }

    if (f3dlbevri.is_open()) { f3dlbevri.close(); }
    else std::cout << "Unable to open error velocity input RESTART file!" << std::endl;

    fname = filename3dlberrorvelocitiesrestartinput;
    f3dlbevri.open(fname.c_str());

    f3dlbevri >> step;
    f3dlbevri.ignore();
    std::string variables[3];
    f3dlbevri >> variables[0] >> variables[1] >> variables[2];

    for (INDEX_TYPE k = startnodez; k <= endnodez; k++) {
        for (INDEX_TYPE j = startnodey; j <= endnodey; j++) {
            for (INDEX_TYPE i = startnodex; i <= endnodex; i++) {
                ixyz = i + j * Nnx + k * Nnx * Nny;
                f3dlbevri >> vxerr[ixyz] >> vyerr[ixyz] >> vzerr[ixyz];
            }
        }
    }

    f3dlbevri.close();
}

template<typename VALUE_TYPE, typename INDEX_TYPE>
void restartInput3DLBForces(
    std::string &filename3dlbforcesrestartinput,
    std::ifstream &f3dlbforceri,
    INDEX_TYPE &Ncx,
    INDEX_TYPE &Ncy,
    INDEX_TYPE &Ncz,
    INDEX_TYPE &Nnx,
    INDEX_TYPE &Nny,
    INDEX_TYPE &Nnz,
    INDEX_TYPE &periodicinx,
    INDEX_TYPE &periodiciny,
    INDEX_TYPE &periodicinz,
    INDEX_TYPE &step,
    VALUE_TYPE *forcex,
    VALUE_TYPE *forcey,
    VALUE_TYPE *forcez)
{
    std::string fname;
    fname = filename3dlbforcesrestartinput;
    f3dlbforceri.open(fname.c_str());

    INDEX_TYPE ixyz;

    INDEX_TYPE startnodex, endnodex;
    if (periodicinx == 0) { startnodex = 0; endnodex = Nnx - 1; }
    else                  { startnodex = 1; endnodex = Nnx - 2; }

    INDEX_TYPE startnodey, endnodey;
    if (periodiciny == 0) { startnodey = 0; endnodey = Nny - 1; }
    else                  { startnodey = 1; endnodey = Nny - 2; }

    INDEX_TYPE startnodez, endnodez;
    if (periodicinz == 0) { startnodez = 0; endnodez = Nnz - 1; }
    else                  { startnodez = 1; endnodez = Nnz - 2; }

    if (f3dlbforceri.is_open()) { f3dlbforceri.close(); }
    else std::cout << "Unable to open force input RESTART file!" << std::endl;

    fname = filename3dlbforcesrestartinput;
    f3dlbforceri.open(fname.c_str());

    f3dlbforceri >> step;
    f3dlbforceri.ignore();
    std::string variables[3];
    f3dlbforceri >> variables[0] >> variables[1] >> variables[2];

    for (INDEX_TYPE k = startnodez; k <= endnodez; k++) {
        for (INDEX_TYPE j = startnodey; j <= endnodey; j++) {
            for (INDEX_TYPE i = startnodex; i <= endnodex; i++) {
                ixyz = i + j * Nnx + k * Nnx * Nny;
                f3dlbforceri >> forcex[ixyz] >> forcey[ixyz] >> forcez[ixyz];
            }
        }
    }

    f3dlbforceri.close();
}

template<typename VALUE_TYPE, typename INDEX_TYPE>
void restartInput3DLBVelocities(
    std::string &filename3dlbvelocitiesrestartinput,
    std::ifstream &f3dlbvri,
    INDEX_TYPE &Ncx,
    INDEX_TYPE &Ncy,
    INDEX_TYPE &Ncz,
    INDEX_TYPE &Nnx,
    INDEX_TYPE &Nny,
    INDEX_TYPE &Nnz,
    INDEX_TYPE &periodicinx,
    INDEX_TYPE &periodiciny,
    INDEX_TYPE &periodicinz,
    INDEX_TYPE &step,
    VALUE_TYPE *vx,
    VALUE_TYPE *vy,
    VALUE_TYPE *vz)
{
    std::string fname;
    fname = filename3dlbvelocitiesrestartinput;
    f3dlbvri.open(fname.c_str());

    INDEX_TYPE ixyz;

    INDEX_TYPE startnodex, endnodex;
    if (periodicinx == 0) { startnodex = 0; endnodex = Nnx - 1; }
    else                  { startnodex = 1; endnodex = Nnx - 2; }

    INDEX_TYPE startnodey, endnodey;
    if (periodiciny == 0) { startnodey = 0; endnodey = Nny - 1; }
    else                  { startnodey = 1; endnodey = Nny - 2; }

    INDEX_TYPE startnodez, endnodez;
    if (periodicinz == 0) { startnodez = 0; endnodez = Nnz - 1; }
    else                  { startnodez = 1; endnodez = Nnz - 2; }

    if (f3dlbvri.is_open()) { f3dlbvri.close(); }
    else std::cout << "Unable to open velocity input RESTART file!" << std::endl;

    fname = filename3dlbvelocitiesrestartinput;
    f3dlbvri.open(fname.c_str());

    f3dlbvri >> step;
    f3dlbvri.ignore();
    std::string variables[3];
    f3dlbvri >> variables[0] >> variables[1] >> variables[2];

    for (INDEX_TYPE k = startnodez; k <= endnodez; k++) {
        for (INDEX_TYPE j = startnodey; j <= endnodey; j++) {
            for (INDEX_TYPE i = startnodex; i <= endnodex; i++) {
                ixyz = i + j * Nnx + k * Nnx * Nny;
                f3dlbvri >> vx[ixyz] >> vy[ixyz] >> vz[ixyz];
            }
        }
    }

    f3dlbvri.close();
}

template<typename VALUE_TYPE, typename INDEX_TYPE>
void restartInput3DLBf(
    std::string &filename3dlbfrestartinput,
    std::ifstream &f3dlbfri,
    INDEX_TYPE &Ncx,
    INDEX_TYPE &Ncy,
    INDEX_TYPE &Ncz,
    INDEX_TYPE &Nnx,
    INDEX_TYPE &Nny,
    INDEX_TYPE &Nnz,
    INDEX_TYPE &periodicinx,
    INDEX_TYPE &periodiciny,
    INDEX_TYPE &periodicinz,
    INDEX_TYPE &step,
    VALUE_TYPE *f)
{
    std::string fname;
    fname = filename3dlbfrestartinput;
    f3dlbfri.open(fname.c_str());

    INDEX_TYPE ixyz, index;

    INDEX_TYPE startnodex, endnodex;
    if (periodicinx == 0) { startnodex = 0; endnodex = Nnx - 1; }
    else                  { startnodex = 1; endnodex = Nnx - 2; }

    INDEX_TYPE startnodey, endnodey;
    if (periodiciny == 0) { startnodey = 0; endnodey = Nny - 1; }
    else                  { startnodey = 1; endnodey = Nny - 2; }

    INDEX_TYPE startnodez, endnodez;
    if (periodicinz == 0) { startnodez = 0; endnodez = Nnz - 1; }
    else                  { startnodez = 1; endnodez = Nnz - 2; }

    if (f3dlbfri.is_open()) { f3dlbfri.close(); }
    else std::cout << "Unable to open f input RESTART file!" << std::endl;

    fname = filename3dlbfrestartinput;
    f3dlbfri.open(fname.c_str());

    f3dlbfri >> step;
    f3dlbfri.ignore();
    std::string variables[19];
    f3dlbfri
        >> variables[0] >> variables[1] >> variables[2]
        >> variables[3] >> variables[4] >> variables[5]
        >> variables[6] >> variables[7] >> variables[8]
        >> variables[9] >> variables[10] >> variables[11]
        >> variables[12] >> variables[13] >> variables[14]
        >> variables[15] >> variables[16] >> variables[17]
        >> variables[18];

    for (INDEX_TYPE k = startnodez; k <= endnodez; k++) {
        for (INDEX_TYPE j = startnodey; j <= endnodey; j++) {
            for (INDEX_TYPE i = startnodex; i <= endnodex; i++) {
                ixyz = i + j * Nnx + k * Nnx * Nny;
                for (INDEX_TYPE l = 0; l < 19; l++) {
                    index = ixyz + l * Nnx * Nny * Nnz;
                    f3dlbfri >> f[index];
                }
            }
        }
    }

    f3dlbfri.close();
}

template<typename VALUE_TYPE, typename INDEX_TYPE>
void restartOutput3DLBDensities(
    std::string &filename3dlbdensitiesrestartoutput,
    std::ofstream &f3dlbrhoro,
    INDEX_TYPE &Ncx,
    INDEX_TYPE &Ncy,
    INDEX_TYPE &Ncz,
    INDEX_TYPE &Nnx,
    INDEX_TYPE &Nny,
    INDEX_TYPE &Nnz,
    INDEX_TYPE &periodicinx,
    INDEX_TYPE &periodiciny,
    INDEX_TYPE &periodicinz,
    INDEX_TYPE &step,
    VALUE_TYPE *rho)
{
    std::string fname;
    fname = filename3dlbdensitiesrestartoutput;
    f3dlbrhoro.open(fname.c_str(), std::ios::out | std::ios::app);

    INDEX_TYPE ixyz;

    INDEX_TYPE startnodex, endnodex;
    if (periodicinx == 0) { startnodex = 0; endnodex = Nnx - 1; }
    else                  { startnodex = 1; endnodex = Nnx - 2; }

    INDEX_TYPE startnodey, endnodey;
    if (periodiciny == 0) { startnodey = 0; endnodey = Nny - 1; }
    else                  { startnodey = 1; endnodey = Nny - 2; }

    INDEX_TYPE startnodez, endnodez;
    if (periodicinz == 0) { startnodez = 0; endnodez = Nnz - 1; }
    else                  { startnodez = 1; endnodez = Nnz - 2; }

    if (f3dlbrhoro.is_open()) { f3dlbrhoro.close(); }
    else std::cout << "Unable to open density output RESTART file!" << std::endl;

    fname = filename3dlbdensitiesrestartoutput;
    f3dlbrhoro.open(fname.c_str(), std::ios::out | std::ios::app);

    f3dlbrhoro << step << "\n" << "rho";

    for (INDEX_TYPE k = startnodez; k <= endnodez; k++) {
        for (INDEX_TYPE j = startnodey; j <= endnodey; j++) {
            for (INDEX_TYPE i = startnodex; i <= endnodex; i++) {
                ixyz = i + j * Nnx + k * Nnx * Nny;
                f3dlbrhoro << std::setprecision(30) << "\n" << rho[ixyz];
            }
        }
    }

    f3dlbrhoro.close();
}

template<typename VALUE_TYPE, typename INDEX_TYPE>
void restartOutput3DLBErrorVelocities(
    std::string &filename3dlberrorvelocitiesrestartoutput,
    std::ofstream &f3dlbevro,
    INDEX_TYPE &Ncx,
    INDEX_TYPE &Ncy,
    INDEX_TYPE &Ncz,
    INDEX_TYPE &Nnx,
    INDEX_TYPE &Nny,
    INDEX_TYPE &Nnz,
    INDEX_TYPE &periodicinx,
    INDEX_TYPE &periodiciny,
    INDEX_TYPE &periodicinz,
    INDEX_TYPE &step,
    VALUE_TYPE *vxerr,
    VALUE_TYPE *vyerr,
    VALUE_TYPE *vzerr)
{
    std::string fname;
    fname = filename3dlberrorvelocitiesrestartoutput;
    f3dlbevro.open(fname.c_str(), std::ios::out | std::ios::app);

    INDEX_TYPE ixyz;

    INDEX_TYPE startnodex, endnodex;
    if (periodicinx == 0) { startnodex = 0; endnodex = Nnx - 1; }
    else                  { startnodex = 1; endnodex = Nnx - 2; }

    INDEX_TYPE startnodey, endnodey;
    if (periodiciny == 0) { startnodey = 0; endnodey = Nny - 1; }
    else                  { startnodey = 1; endnodey = Nny - 2; }

    INDEX_TYPE startnodez, endnodez;
    if (periodicinz == 0) { startnodez = 0; endnodez = Nnz - 1; }
    else                  { startnodez = 1; endnodez = Nnz - 2; }

    if (f3dlbevro.is_open()) { f3dlbevro.close(); }
    else std::cout << "Unable to open error velocity output RESTART file!" << std::endl;

    fname = filename3dlberrorvelocitiesrestartoutput;
    f3dlbevro.open(fname.c_str(), std::ios::out | std::ios::app);

    f3dlbevro << step << "\n" << "vx vy vz";

    for (INDEX_TYPE k = startnodez; k <= endnodez; k++) {
        for (INDEX_TYPE j = startnodey; j <= endnodey; j++) {
            for (INDEX_TYPE i = startnodex; i <= endnodex; i++) {
                ixyz = i + j * Nnx + k * Nnx * Nny;
                f3dlbevro << std::setprecision(30)
                          << "\n"
                          << vxerr[ixyz] << " "
                          << vyerr[ixyz] << " "
                          << vzerr[ixyz];
            }
        }
    }

    f3dlbevro.close();
}

template<typename VALUE_TYPE, typename INDEX_TYPE>
void restartOutput3DLBForces(
    std::string &filename3dlbforcesrestartoutput,
    std::ofstream &f3dlbforcero,
    INDEX_TYPE &Ncx,
    INDEX_TYPE &Ncy,
    INDEX_TYPE &Ncz,
    INDEX_TYPE &Nnx,
    INDEX_TYPE &Nny,
    INDEX_TYPE &Nnz,
    INDEX_TYPE &periodicinx,
    INDEX_TYPE &periodiciny,
    INDEX_TYPE &periodicinz,
    INDEX_TYPE &step,
    VALUE_TYPE *forcex,
    VALUE_TYPE *forcey,
    VALUE_TYPE *forcez)
{
    std::string fname;
    fname = filename3dlbforcesrestartoutput;
    f3dlbforcero.open(fname.c_str(), std::ios::out | std::ios::app);

    INDEX_TYPE ixyz;

    INDEX_TYPE startnodex, endnodex;
    if (periodicinx == 0) { startnodex = 0; endnodex = Nnx - 1; }
    else                  { startnodex = 1; endnodex = Nnx - 2; }

    INDEX_TYPE startnodey, endnodey;
    if (periodiciny == 0) { startnodey = 0; endnodey = Nny - 1; }
    else                  { startnodey = 1; endnodey = Nny - 2; }

    INDEX_TYPE startnodez, endnodez;
    if (periodicinz == 0) { startnodez = 0; endnodez = Nnz - 1; }
    else                  { startnodez = 1; endnodez = Nnz - 2; }

    if (f3dlbforcero.is_open()) { f3dlbforcero.close(); }
    else std::cout << "Unable to open force output RESTART file!" << std::endl;

    fname = filename3dlbforcesrestartoutput;
    f3dlbforcero.open(fname.c_str(), std::ios::out | std::ios::app);

    f3dlbforcero << step << "\n" << "forcex forcey forcez";

    for (INDEX_TYPE k = startnodez; k <= endnodez; k++) {
        for (INDEX_TYPE j = startnodey; j <= endnodey; j++) {
            for (INDEX_TYPE i = startnodex; i <= endnodex; i++) {
                ixyz = i + j * Nnx + k * Nnx * Nny;
                f3dlbforcero << std::setprecision(30)
                             << "\n"
                             << forcex[ixyz] << " "
                             << forcey[ixyz] << " "
                             << forcez[ixyz];
            }
        }
    }

    f3dlbforcero.close();
}

template<typename VALUE_TYPE, typename INDEX_TYPE>
void restartOutput3DLBVelocities(
    std::string &filename3dlbvelocitiesrestartoutput,
    std::ofstream &f3dlbvro,
    INDEX_TYPE &Ncx,
    INDEX_TYPE &Ncy,
    INDEX_TYPE &Ncz,
    INDEX_TYPE &Nnx,
    INDEX_TYPE &Nny,
    INDEX_TYPE &Nnz,
    INDEX_TYPE &periodicinx,
    INDEX_TYPE &periodiciny,
    INDEX_TYPE &periodicinz,
    INDEX_TYPE &step,
    VALUE_TYPE *vx,
    VALUE_TYPE *vy,
    VALUE_TYPE *vz)
{
    std::string fname;
    fname = filename3dlbvelocitiesrestartoutput;
    f3dlbvro.open(fname.c_str(), std::ios::out | std::ios::app);

    INDEX_TYPE ixyz;

    INDEX_TYPE startnodex, endnodex;
    if (periodicinx == 0) { startnodex = 0; endnodex = Nnx - 1; }
    else                  { startnodex = 1; endnodex = Nnx - 2; }

    INDEX_TYPE startnodey, endnodey;
    if (periodiciny == 0) { startnodey = 0; endnodey = Nny - 1; }
    else                  { startnodey = 1; endnodey = Nny - 2; }

    INDEX_TYPE startnodez, endnodez;
    if (periodicinz == 0) { startnodez = 0; endnodez = Nnz - 1; }
    else                  { startnodez = 1; endnodez = Nnz - 2; }

    if (f3dlbvro.is_open()) { f3dlbvro.close(); }
    else std::cout << "Unable to open velocity output RESTART file!" << std::endl;

    fname = filename3dlbvelocitiesrestartoutput;
    f3dlbvro.open(fname.c_str(), std::ios::out | std::ios::app);

    f3dlbvro << step << "\n" << "vx vy vz";

    for (INDEX_TYPE k = startnodez; k <= endnodez; k++) {
        for (INDEX_TYPE j = startnodey; j <= endnodey; j++) {
            for (INDEX_TYPE i = startnodex; i <= endnodex; i++) {
                ixyz = i + j * Nnx + k * Nnx * Nny;
                f3dlbvro << std::setprecision(30)
                         << "\n"
                         << vx[ixyz] << " "
                         << vy[ixyz] << " "
                         << vz[ixyz];
            }
        }
    }

    f3dlbvro.close();
}

template<typename VALUE_TYPE, typename INDEX_TYPE>
void restartOutput3DLBf(
    std::string &filename3dlbfrestartoutput,
    std::ofstream &f3dlbfro,
    INDEX_TYPE &Ncx,
    INDEX_TYPE &Ncy,
    INDEX_TYPE &Ncz,
    INDEX_TYPE &Nnx,
    INDEX_TYPE &Nny,
    INDEX_TYPE &Nnz,
    INDEX_TYPE &periodicinx,
    INDEX_TYPE &periodiciny,
    INDEX_TYPE &periodicinz,
    INDEX_TYPE &step,
    VALUE_TYPE *f)
{
    std::string fname;
    fname = filename3dlbfrestartoutput;
    f3dlbfro.open(fname.c_str(), std::ios::out | std::ios::app);

    INDEX_TYPE ixyz, index;

    INDEX_TYPE startnodex, endnodex;
    if (periodicinx == 0) { startnodex = 0; endnodex = Nnx - 1; }
    else                  { startnodex = 1; endnodex = Nnx - 2; }

    INDEX_TYPE startnodey, endnodey;
    if (periodiciny == 0) { startnodey = 0; endnodey = Nny - 1; }
    else                  { startnodey = 1; endnodey = Nny - 2; }

    INDEX_TYPE startnodez, endnodez;
    if (periodicinz == 0) { startnodez = 0; endnodez = Nnz - 1; }
    else                  { startnodez = 1; endnodez = Nnz - 2; }

    if (f3dlbfro.is_open()) { f3dlbfro.close(); }
    else std::cout << "Unable to open f output RESTART file!" << std::endl;

    fname = filename3dlbfrestartoutput;
    f3dlbfro.open(fname.c_str(), std::ios::out | std::ios::app);

    f3dlbfro << step << "\n"
             << "f0 f1 f2 f3 f4 f5 f6 f7 f8 f9 "
             << "f10 f11 f12 f13 f14 f15 f16 f17 f18";

    for (INDEX_TYPE k = startnodez; k <= endnodez; k++) {
        for (INDEX_TYPE j = startnodey; j <= endnodey; j++) {
            for (INDEX_TYPE i = startnodex; i <= endnodex; i++) {
                ixyz = i + j * Nnx + k * Nnx * Nny;
                f3dlbfro << "\n";
                for (INDEX_TYPE l = 0; l < 19; l++) {
                    index = ixyz + l * Nnx * Nny * Nnz;
                    if (l == 18) f3dlbfro << std::setprecision(30) << f[index];
                    else         f3dlbfro << std::setprecision(30) << f[index] << " ";
                }
            }
        }
    }

    f3dlbfro.close();
}
