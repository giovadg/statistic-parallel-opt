#include <vector>
#include <tuple>
#include <random>
#include <chrono>
#include <iostream>
#include <cstdlib>
#include <boost/random.hpp>
#include <unistd.h>
#include <fstream>

namespace generation{
void interface_vectors_generation(string path,int n_vect, int n, vector<vector<double>>& x_tot,
                                    vector<vector<double>>& roll_av_ser, vector<vector<double>>& roll_av_pll,
                                    vector<vector<vector<double>>>& roll_corr_ser, vector<vector<vector<double>>>& roll_corr_pll );
}