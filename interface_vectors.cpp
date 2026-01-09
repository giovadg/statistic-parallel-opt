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
#include <cstdint>

using namespace std; 

namespace in_out{





    

void read_bin(string path, vector<vector<double>>& x_tot,
               vector<vector<double>>& roll_av_ser, vector<vector<double>>& roll_av_pll,
               vector<vector<double>>& roll_var_ser, vector<vector<double>>& roll_var_pll,
               vector<vector<vector<double>>>& roll_corr_ser, vector<vector<vector<double>>>& roll_corr_pll)   // contiguo: X[i*ncols + j]
{
    vector<double> X;
    uint64_t r, c;

    ifstream input(path, ios::binary);
    if (!input.is_open()) throw runtime_error("cannot open file");

    input.read(reinterpret_cast<char*>(&r), sizeof(uint64_t));
    input.read(reinterpret_cast<char*>(&c), sizeof(uint64_t));
    if (!input.good()) throw runtime_error("bad header");

    size_t n_ele = (size_t)r; size_t n_vect = (size_t)c;

    X.resize(n_ele * n_vect);
    x_tot.resize(n_vect, vector<double>(n_ele));
    roll_av_ser.resize(n_vect, vector<double>(n_ele));
    roll_av_pll.resize(n_vect, vector<double>(n_ele));
    roll_var_ser.resize(n_vect, vector<double>(n_ele));
    roll_var_pll.resize(n_vect, vector<double>(n_ele));
    roll_corr_ser.resize(n_vect, vector<vector<double>>(n_vect, vector<double>(n_ele)));
    roll_corr_pll.resize(n_vect, vector<vector<double>>(n_vect, vector<double>(n_ele)));


    input.read(reinterpret_cast<char*>(X.data()), X.size() * sizeof(double));
    if (!input.good()) throw runtime_error("bad data");

    for (size_t i=0;i<n_ele;++i){
        for (size_t j=0;j<n_vect;++j){
            x_tot[j][i] = X[i*n_vect + j];
        }
    }
}


void read_csv(string path, vector<vector<double>>& x_tot,
               vector<vector<double>>& roll_av_ser, vector<vector<double>>& roll_av_pll,
               vector<vector<double>>& roll_var_ser, vector<vector<double>>& roll_var_pll,
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
    roll_var_ser.resize(ncols, vector<double>(nrows));
    roll_var_pll.resize(ncols, vector<double>(nrows));
    roll_corr_ser.resize(ncols, vector<vector<double>>(ncols, vector<double>(nrows)));
    roll_corr_pll.resize(ncols, vector<vector<double>>(ncols, vector<double>(nrows)));

    return;
}

void generate_vectors(int n_vect, int n, vector<vector<double>>& x_tot,
                vector<vector<double>>& roll_av_ser, vector<vector<double>>& roll_av_pll,
                vector<vector<double>>& roll_var_ser, vector<vector<double>>& roll_var_pll,
                vector<vector<vector<double>>>& roll_corr_ser, vector<vector<vector<double>>>& roll_corr_pll ){

    x_tot.resize(n_vect, vector<double>(n));
    roll_av_ser.resize(n_vect, vector<double>(n));
    roll_av_pll.resize(n_vect, vector<double>(n));
    roll_var_ser.resize(n_vect, vector<double>(n));
    roll_var_pll.resize(n_vect, vector<double>(n));
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


void read_file(string path, vector<vector<double>>& x_tot,
               vector<vector<double>>& roll_av_ser, vector<vector<double>>& roll_av_pll,
               vector<vector<double>>& roll_var_ser, vector<vector<double>>& roll_var_pll,
               vector<vector<vector<double>>>& roll_corr_ser, vector<vector<vector<double>>>& roll_corr_pll ){
    
    if (path.contains(".bin")) {
        read_bin(path, x_tot, roll_av_ser, roll_av_pll, roll_var_ser, roll_var_pll, roll_corr_ser, roll_corr_pll);
    }else if(path.contains(".csv")){
        read_csv(path, x_tot, roll_av_ser, roll_av_pll, roll_var_ser, roll_var_pll, roll_corr_ser, roll_corr_pll);
    }else{
        throw runtime_error("not a good input file extension");
    }
    return ;
    }

void interface_vectors_generation(string path,int n_vect, int n, vector<vector<double>>& x_tot,
                                    vector<vector<double>>& roll_av_ser, vector<vector<double>>& roll_av_pll,
                                    vector<vector<double>>& roll_var_ser, vector<vector<double>>& roll_var_pll,
                                    vector<vector<vector<double>>>& roll_corr_ser, vector<vector<vector<double>>>& roll_corr_pll ){

    if (path != "none"){
        cout << " reading from file\n";
        read_file(path, x_tot, roll_av_ser, roll_av_pll, roll_var_ser, roll_var_pll, roll_corr_ser, roll_corr_pll);
    }else{
        cout << " generating vectors\n";
        generate_vectors(n_vect, n, x_tot, roll_av_ser, roll_var_ser, roll_var_pll, roll_av_pll, roll_corr_ser, roll_corr_pll);
    }
    return;
}


void save_correlation(const vector<vector<vector<double>>>& corr,
                 const string& fname)
{
    int  n_vect = corr.size();
    size_t n_el = corr[0][0].size();

    std::ofstream out(fname, std::ios::binary);

    out.write(reinterpret_cast<char*>(&n_vect), sizeof(int));
    out.write(reinterpret_cast<char*>(&n_vect), sizeof(int));
    out.write(reinterpret_cast<char*>(&n_el), sizeof(size_t));

    for (int ii = 0; ii < n_vect; ii++){
        for (int jj = 0; jj < n_vect; jj++){
            out.write(reinterpret_cast<const char*>(corr[ii][jj].data()),
                    n_el * sizeof(double));
        }
    }
}


};
