#include <cstdlib>
#include "kernels.hpp"
#include <cmath>
#include <limits>
#include <algorithm>
#include <vector>
#include <numeric>
#include <pthread.h>
#include <iostream>

using namespace std; 

namespace kernels {


void rolling_mean_serial_exec(const vector<double> &arr_in, vector<double> &arr_out,
                         size_t &w, int start_index, int end_index){

    if (end_index == -1) end_index = arr_in.size();

    // 1. compute the first average
    double sum = std::accumulate(arr_in.begin()+start_index, arr_in.begin()+start_index+w, 0.0);
    arr_out[start_index] = sum/w;

    // 3. Computes the other averages using DP
    //    Iterate from the starting index + 1 until the last available point   
    for (int ii=start_index+1; ii<end_index-w;ii++){
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
    int num_threads;
};

// single thread execution function
void* rolling_mean_single_thr_exe(void* arg){

    // 1. cast the void pointer to its structure
    // ThreadArgs* state = *(ThreadArgs*)arg;
    ThreadArgs* state = static_cast<ThreadArgs*>(arg);

    rolling_mean_serial_exec(*state->arr_in, *state->arr_out, state->w, state->start_index, state->end_index);
    
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
        args[jj].end_index   = (jj == num_threads - 1) ? arr_in.size() : (jj + 1) * chunk+w;        

        pthread_create(&th[jj], NULL, &rolling_mean_single_thr_exe, &args[jj]);
    }

    // wait for threads to finish
    for (int jj = 0; jj < num_threads; jj++) {
        pthread_join(th[jj], NULL);
    }

    return;
}

void* rolling_mean_parallel_interface(void* arg_inp){

    ThreadArgs* state = static_cast<ThreadArgs*>(arg_inp);

    int num_threads = state->num_threads;
    int chunk = (*state->arr_in).size()/(num_threads);

    pthread_t th[num_threads];

    // Creation of the sing thread function arguments
    ThreadArgs args[num_threads]; 

    // Each thread has its own arguments
    for(int jj=0; jj<num_threads;jj++){
        args[jj].arr_in  = state->arr_in;
        args[jj].arr_out = state->arr_out;
        args[jj].w       = state->w;
        args[jj].num_threads  = num_threads;
        
        // Range division
        args[jj].start_index = jj * chunk;
        args[jj].end_index = (jj == num_threads - 1) ? (*state->arr_in).size() : (jj + 1) * chunk+state->w;        

        pthread_create(&th[jj], NULL, &rolling_mean_single_thr_exe, &args[jj]);
    }

    // wait for threads to finish
    for (int jj = 0; jj < num_threads; jj++) {
        pthread_join(th[jj], NULL);
    }

    return nullptr;
}



void* rolling_mean_serial_interface(void* arg_inp){

    ThreadArgs* state = static_cast<ThreadArgs*>(arg_inp);

    rolling_mean_serial_exec(*state->arr_in, *state->arr_out, state->w);
    
    return nullptr;
}

void rolling_mean_parallel_inputs(vector<vector<double>> &arrs_in, vector<vector<double>> &arrs_out,
                                     size_t &w, int num_threads, bool nested_threads){


    pthread_t th[arrs_in.size()];

    // Creation of the sing thread function arguments
    ThreadArgs args[arrs_in.size()]; 

    // Each thread has its own arguments
    // for(int jj=0; jj<arrs_in.size();jj++){
    for(int jj=0; jj<arrs_in.size();jj++){
        args[jj].arr_in  = &arrs_in[jj];
        args[jj].arr_out = &arrs_out[jj];
        args[jj].w       = w;
        args[jj].num_threads = std::max(1,int(num_threads/arrs_in.size()));
        
        // Range division
        args[jj].start_index = 0;
        args[jj].end_index   = arrs_in[jj].size();        

        if(nested_threads){
            if (jj==0) printf("using a total of %d threads for par vect input and par vect treat \n \n", args[jj].num_threads * arrs_in.size());
            pthread_create(&th[jj], NULL, &rolling_mean_parallel_interface, &args[jj]);
        }else{         
            pthread_create(&th[jj], NULL, &rolling_mean_serial_interface, &args[jj]);
        }
    }

    // wait for threads to finish
    for (int jj = 0; jj < arrs_in.size(); jj++) {
        pthread_join(th[jj], NULL);
    }

    return;
}


}