#include <cstddef>
#include <cstdlib>
#include <vector>

using namespace std; 

namespace kernels {

void rolling_mean_serial(const vector<double> &arr_in, vector<double> &arr_out, size_t &w);

// struct ThreadArgs {
//     const vector<double>* arr_in;
//     vector<double>* arr_out;
//     size_t w;
//     int start_index;
//     int end_index;
// };

void rolling_mean_parallel(vector<double> &arr_in, vector<double> &arr_out, size_t &w,  int num_threads);


void rolling_mean_parallel_inputs(vector<vector<double>> &arrs_in, vector<vector<double>> &arrs_out,
                                     size_t &w, int num_threads, bool nested_threads);



};