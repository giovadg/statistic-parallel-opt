#include <cstddef>
#include <cstdlib>
#include <vector>
#include <unistd.h>
#include <iostream>

using namespace std; 

namespace kernels {

void rolling_mean_exec(const vector<vector<double>> &arrs_in, vector<vector<double>> &arr_out, size_t &w, int start_index=0,
                                 int end_index=-1, int vect_start=0, int vect_end=-1);

void rolling_var_exec(const vector<vector<double>> &arrs_in, vector<vector<double>> &arr_out, size_t &w, int start_index=0,
                                 int end_index=-1, int vect_start=0, int vect_end=-1);
void rolling_stat_parallel(const vector<vector<double>> &arr_in, vector<vector<double>> &arr_out, string method, size_t &w,  int num_threads);

void rolling_stat_parallel_nested(const vector<vector<double>> &arrs_in, vector<vector<double>> &arrs_out, string method,
                                     size_t &w, int num_threads, bool nested_threads);


void rolling_corr_parallel(const vector<vector<double>> &vect, 
                        vector<vector<double>> &vect_mean, vector<vector<double>> &vect_std,
                        vector<vector<vector<double>>> &arr_out, size_t &w, int num_threads);

void rolling_mean_corr_exec(const vector<vector<double>> &tuple_vect, 
                        vector<vector<double>> &tuple_mean, 
                        vector<double> &arr_out, size_t &w, int start_index=0, int end_index=-1);


void rolling_mean_corr_exec_mv(const vector<vector<double>> &vect, 
                        vector<vector<double>> &vect_mean, 
                        vector<vector<double>> &vect_var,
                        vector<vector<vector<double>>> &arr_out, size_t &w, int start_index=0, int end_index=-1);

};