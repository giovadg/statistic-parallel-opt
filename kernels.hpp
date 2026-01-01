#include <cstddef>
#include <cstdlib>
#include <vector>

using namespace std; 

namespace kernels {

void rolling_mean_exec(const vector<double> &arr_in, vector<double> &arr_out, size_t &w, int start_index=0, int end_index=-1);

void rolling_mean_parallel(vector<double> &arr_in, vector<double> &arr_out, size_t &w,  int num_threads);

void rolling_mean_parallel_inputs(vector<vector<double>> &arrs_in, vector<vector<double>> &arrs_out,
                                     size_t &w, int num_threads, bool nested_threads);


void rolling_corr_parallel(const tuple<vector<double>, vector<double>> &tuple_vect, 
                        const tuple<vector<double>, vector<double>> &tuple_mean, 
                        vector<double> &arr_out, size_t &w, int num_threads);
void rolling_corr_exec(const tuple<vector<double>, vector<double>> &tuple_vect, 
                        const tuple<vector<double>, vector<double>> &tuple_mean, 
                        vector<double> &arr_out, size_t &w, int start_index=0, int end_index=-1);
};