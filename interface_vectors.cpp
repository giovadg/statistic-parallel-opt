#include <vector>
#include <string>
#include <sstream> // Fondamentale per stringstream
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <boost/random.hpp>
#include <chrono>
#include <unistd.h>
#include <cstdlib>

using namespace std; 

namespace generation{

void read_file(string path, vector<vector<double>>& x_tot,
                vector<vector<double>>& roll_av_ser, vector<vector<double>>& roll_av_pll,
                vector<vector<double>>& roll_corr_ser, vector<vector<double>>& roll_corr_pll ){
    int ii(0), n_line;
    ifstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("cannot open file.");
    } else {
        std::string line, cell;

        // leggi prima riga
        if (!std::getline(file, line)) throw std::runtime_error("empty file");

        // conta colonne
        size_t ncols = 0;
        {
            std::stringstream ss(line);
            while (std::getline(ss, cell, ';')) ++ncols;
        }
        x_tot.resize(ncols);

        {
            std::stringstream ss(line);
            size_t ii = 0;
            while (std::getline(ss, cell, ';')) {
                x_tot[ii++].push_back(std::stod(cell));
            }
        }

        while (std::getline(file, line)) {
            std::stringstream ss(line);
            size_t ii = 0;
            while (std::getline(ss, cell, ';')) {
                x_tot[ii++].push_back(std::stod(cell));
            }
            if (ii != ncols) throw std::runtime_error("inconsistent number of columns");
        }
    }

    size_t ncols = x_tot.size();
    size_t nrows = x_tot[0].size();

    roll_av_ser.resize(ncols, std::vector<double>(nrows));
    roll_av_pll.resize(ncols, std::vector<double>(nrows));
    roll_corr_ser.resize(ncols, std::vector<double>(nrows));
    roll_corr_pll.resize(ncols, std::vector<double>(nrows));

    return;
}

void generate_vectors(int n,vector<vector<double>>& x_tot,
                vector<vector<double>>& roll_av_ser, vector<vector<double>>& roll_av_pll,
                vector<vector<double>>& roll_corr_ser, vector<vector<double>>& roll_corr_pll ){

    x_tot.resize(2, std::vector<double>(n));
    roll_av_ser.resize(2, std::vector<double>(n));
    roll_av_pll.resize(2, std::vector<double>(n));
    roll_corr_ser.resize(2, std::vector<double>(n));
    roll_corr_pll.resize(2, std::vector<double>(n));

    // Random number generators: 2 Gaussians, with different mean and std. Different seeds.
    boost::mt19937 rng1(123);
    boost::mt19937 rng2(100);
    boost::normal_distribution<> initial_distribution1(0,1);
    boost::normal_distribution<> initial_distribution2(1,2);

    boost::variate_generator<boost::mt19937&, boost::normal_distribution<>> 
    initial_velocities_generator1(rng1,initial_distribution1);

    boost::variate_generator<boost::mt19937&, boost::normal_distribution<>> 
    initial_velocities_generator2(rng2,initial_distribution2);

    for (size_t i = 0; i < n; i++){
    x_tot[0][i] = initial_velocities_generator1();
    x_tot[1][i] = initial_velocities_generator2();
    }




    return;
}


void interface_vectors_generation(string path,int n, vector<vector<double>>& x_tot,
                                    vector<vector<double>>& roll_av_ser, vector<vector<double>>& roll_av_pll,
                                    vector<vector<double>>& roll_corr_ser, vector<vector<double>>& roll_corr_pll ){

    if (path != "none"){
        cout << " reading from file\n";
        read_file(path, x_tot,
                 roll_av_ser,  roll_av_pll,
                     roll_corr_ser, roll_corr_pll);
    }else{
        cout << " generating vectors\n";
        generate_vectors(n, x_tot,
                       roll_av_ser,  roll_av_pll,
                        roll_corr_ser, roll_corr_pll);


    }
    return;
}

};
