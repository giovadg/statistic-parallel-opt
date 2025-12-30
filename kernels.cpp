#include <cstdlib>
#include "kernels.hpp"
#include <cmath>
#include <limits>
#include <algorithm>
#include <vector>
#include <numeric>
#include <pthread.h>
#include <iostream>

namespace kernels {
using namespace std; 


void rolling_mean_serial(vector<double> &arr_in, vector<double> &arr_out, size_t &w){

    // 1. compute the first average
    double sum = std::accumulate(arr_in.begin(), arr_in.begin()+w, 0.0);
    arr_out[0] = sum/w;

    // 3. Computes the other averages using DP
    //    Iterate from the starting index + 1 until the last available point   
    for (int ii=1; ii<arr_in.size()-w;ii++){
        arr_out[ii] = (w*arr_out[ii-1] - arr_in[ii-1] + arr_in[ii+w-1])/w;
    }
    return;
}

// define the structure of each thread input
struct ThreadArgs {
    const vector<double>* arr_in;
    vector<double>* arr_out;
    size_t w;
    int start_index;
    int end_index;
};

// single thread execution function
void* rolling_mean_single_thr_exe(void* arg){
    // 1. cast the void pointer to its structure
    // ThreadArgs* state = *(ThreadArgs*)arg;
    ThreadArgs* state = static_cast<ThreadArgs*>(arg);
    auto init = state->arr_in->begin() + state->start_index;
    
    // 2. compute the first average
    double sum = std::accumulate(init, init + state->w, 0.0);
     (*(state->arr_out))[state->start_index] = sum/state->w;

    // define final point leaving out w elements for the average
    auto final_point = state->end_index - state->w;

    // 3. Computes the other averages using DP
    //    Iterate from the starting index + 1 until the last available point
    for (int ii=state->start_index+1; ii<final_point;ii++){
        (*(state->arr_out))[ii] = (state->w* (*(state->arr_out))[ii-1] 
                                - (*(state->arr_in))[ii-1] + (*(state->arr_in))[ii+state->w-1])/state->w;
    }
    
    return nullptr; 
}



void rolling_mean_parallel(vector<double> &arr_in, vector<double> &arr_out, size_t &w, int num_threads){

    int chunk = arr_in.size()/num_threads;

    pthread_t th[num_threads];

    // Creation of the sing thread function arguments
    ThreadArgs args[num_threads]; 

    // Each thread has its own arguments
    for(int jj=0; jj<num_threads;jj++){
        args[jj].arr_in  = &arr_in;
        args[jj].arr_out = &arr_out;
        args[jj].w       = w;
        
        // Range division
        args[jj].start_index = jj * chunk;
        args[jj].end_index = (jj == num_threads - 1) ? arr_in.size() : (jj + 1) * chunk+w;        

        pthread_create(&th[jj], NULL, &rolling_mean_single_thr_exe, &args[jj]);
    }

    // wait for threads to finish
    for (int jj = 0; jj < num_threads; jj++) {
        pthread_join(th[jj], NULL);
    }

    return;
}

}