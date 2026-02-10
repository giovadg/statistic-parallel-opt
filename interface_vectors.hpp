#include <vector>
#include <tuple>
#include <random>
#include <chrono>
#include <iostream>
#include <cstdlib>
#include <boost/random.hpp>
#include <unistd.h>
#include <fstream>

namespace in_out{
void interface_vectors_generation(string path,int n_vect, int n, vector<vector<double>>& x_tot,
                                    vector<vector<double>>& roll_av_ser, vector<vector<double>>& roll_av_pll,
                                    vector<vector<double>>& roll_var_ser, vector<vector<double>>& roll_var_pll,
                                    vector<vector<vector<double>>>& roll_corr_ser, vector<vector<vector<double>>>& roll_corr_pll );


void save_stat_impl(const vector<vector<double>>& stat,
                 const string& fname);

void save_stat_impl(const vector<vector<vector<double>>>& stat,
                 const string& fname);

void save_stat(const std::vector<std::vector<double>>& stat,
               const std::string& fname);

void save_stat(const std::vector<std::vector<std::vector<double>>>& stat,
               const std::string& fname);


void save_stat(const vector<double>& stat, size_t n_vect, size_t n_el, 
                 const string& fname);
}
