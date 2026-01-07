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
                vector<vector<vector<double>>>& roll_corr_ser, vector<vector<vector<double>>>& roll_corr_pll ){
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

    roll_av_ser.resize(ncols, vector<double>(nrows));
    roll_av_pll.resize(ncols, vector<double>(nrows));
    roll_corr_ser.resize(ncols, vector<vector<double>>(ncols, vector<double>(nrows)));
    roll_corr_pll.resize(ncols, vector<vector<double>>(ncols, vector<double>(nrows)));

    return;
}

void generate_vectors(int n_vect, int n, vector<vector<double>>& x_tot,
                vector<vector<double>>& roll_av_ser, vector<vector<double>>& roll_av_pll,
                vector<vector<vector<double>>>& roll_corr_ser, vector<vector<vector<double>>>& roll_corr_pll ){

    x_tot.resize(n_vect, vector<double>(n));
    roll_av_ser.resize(n_vect, vector<double>(n));
    roll_av_pll.resize(n_vect, vector<double>(n));

    roll_corr_ser.resize(n_vect, vector<vector<double>>(n_vect, vector<double>(n)));
    roll_corr_pll.resize(n_vect, vector<vector<double>>(n_vect, vector<double>(n)));

    // mean/std for each vector (example)
    vector<double> mu(n_vect), sigma(n_vect);
    for (int nv=0; nv<n_vect; nv++){
        mu[nv]    = 0.0 + nv;
        sigma[nv] = 1.0 + nv;
    }

    // different seed for each vector but deterministic-reproducible
    vector<boost::mt19937> rng(n_vect);
    for (int nv=0; nv<n_vect; nv++){
        rng[nv].seed(100 + nv);
    }

    for (int nv=0; nv<n_vect; nv++){
        boost::normal_distribution<> dist(mu[nv], sigma[nv]);
        boost::variate_generator<boost::mt19937&, boost::normal_distribution<> > gen(rng[nv], dist);

        for (size_t i=0; i<n; i++){
            x_tot[nv][i] = gen();
        }
    }

    return;
}


void interface_vectors_generation(string path,int n_vect, int n, vector<vector<double>>& x_tot,
                                    vector<vector<double>>& roll_av_ser, vector<vector<double>>& roll_av_pll,
                                    vector<vector<vector<double>>>& roll_corr_ser, vector<vector<vector<double>>>& roll_corr_pll ){

    if (path != "none"){
        cout << " reading from file\n";
        read_file(path, x_tot,
                 roll_av_ser,  roll_av_pll,
                     roll_corr_ser, roll_corr_pll);
    }else{
        cout << " generating vectors\n";
        generate_vectors(n_vect, n, x_tot,
                       roll_av_ser,  roll_av_pll,
                        roll_corr_ser, roll_corr_pll);


    }
    return;
}

};
