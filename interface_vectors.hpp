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
void interface_vectors_generation(string path,int n, vector<vector<double>>& x_tot,
                                    vector<vector<double>>& roll_av_ser, vector<vector<double>>& roll_av_pll,
                                    vector<vector<double>>& roll_corr_ser, vector<vector<double>>& roll_corr_pll );
}