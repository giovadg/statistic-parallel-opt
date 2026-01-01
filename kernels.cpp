#include <cstdlib>
#include "kernels.hpp"
#include <cmath>
#include <limits>
#include <algorithm>
#include <vector>
#include <tuple>
#include <numeric>
#include <pthread.h>
#include <iostream>

using namespace std; 

namespace kernels {



void rolling_corr_exec(const tuple<vector<double>, vector<double>> &tuple_vect, 
                        const tuple<vector<double>, vector<double>> &tuple_mean, 
                        vector<double> &arr_out, size_t &w, int start_index, int end_index){

    auto& vect1 = std::get<0>(tuple_vect);
    auto& vect2 = std::get<1>(tuple_vect);
    auto& vect1_mean = std::get<0>(tuple_mean);
    auto& vect2_mean = std::get<1>(tuple_mean);
    if (end_index == -1) end_index = min(vect1.size(), vect2.size());
    
    cout <<" "<<end_index<< " \n";
    double corr_NN = 0.0;
    double std1 = 0.0;
    double std2 = 0.0;
    double denom;
    // 1. compute the first corr
    double Sv1v1    = 0.0;
    double Sv2v2    = 0.0;
    double Sv1v2    = 0.0;
    for (int ii = start_index; ii < start_index+w; ii++){
        Sv1v1 += vect1[ii]* vect1[ii];
        Sv1v2 += vect1[ii]* vect2[ii];
        Sv2v2 += vect2[ii]* vect2[ii];
    }
    std1    = Sv1v1/w - vect1_mean[start_index] * vect1_mean[start_index];
    std2    = Sv2v2/w - vect2_mean[start_index] * vect2_mean[start_index];
    corr_NN = Sv1v2/w - vect1_mean[start_index] * vect2_mean[start_index];

    denom = std::sqrt(std1*std2);
    arr_out[start_index] = (denom > 0) ? corr_NN/denom : 0.0;

    // DP part for the rolling window
    for (int ii = start_index+1; ii < end_index-w; ii++){
        Sv1v1 += vect1[ii+w-1]* vect1[ii+w-1] - vect1[ii-1]* vect1[ii-1] ;
        Sv1v2 += vect1[ii+w-1]* vect2[ii+w-1] - vect1[ii-1]* vect2[ii-1] ;
        Sv2v2 += vect2[ii+w-1]* vect2[ii+w-1] - vect2[ii-1]* vect2[ii-1] ;
    
        std1    = Sv1v1/w - vect1_mean[ii] * vect1_mean[ii];
        std2    = Sv2v2/w - vect2_mean[ii] * vect2_mean[ii];
        corr_NN = Sv1v2/w - vect1_mean[ii] * vect2_mean[ii];

        denom = std::sqrt(std1*std2);
        arr_out[ii] = (denom > 0) ? corr_NN/denom : 0.0;

        if (abs(arr_out[ii])>1 ) printf("error in the algorithm, correlation larger than 1.\n");
    }
    return;
}

void rolling_mean_exec(const vector<double> &arr_in, vector<double> &arr_out,
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
struct ThreadArgs_mean {
    const vector<double>* arr_in;
    vector<double>* arr_out;
    size_t w;
    int start_index;
    int end_index;
    int num_threads;
};
struct ThreadArgs_corr {
    const tuple<vector<double>, vector<double>>* tuple_vect;
    const tuple<vector<double>, vector<double>>* tuple_mean;
    vector<double>* arr_out;
    size_t w;
    int start_index;
    int end_index;
    int num_threads;
};


// single thread execution function
void* rolling_mean_single_thr_exe(void* arg){

    // 1. cast the void pointer to its structure
    ThreadArgs_mean* state = static_cast<ThreadArgs_mean*>(arg);

    rolling_mean_exec(*state->arr_in, *state->arr_out, state->w, state->start_index, state->end_index);
    
    return nullptr; 
}

// single thread execution function
void* rolling_corr_single_thr_exe(void* arg){

    // 1. cast the void pointer to its structure
    ThreadArgs_corr* state = static_cast<ThreadArgs_corr*>(arg);

    rolling_corr_exec(*state->tuple_vect, *state->tuple_mean, *state->arr_out, state->w, state->start_index, state->end_index);
    
    return nullptr; 
}

void rolling_mean_parallel(vector<double> &arr_in, vector<double> &arr_out, size_t &w, int num_threads){

    int chunk = arr_in.size()/num_threads;

    pthread_t th[num_threads];

    // Creation of the sing thread function arguments
    ThreadArgs_mean args[num_threads]; 

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

void rolling_corr_parallel(const tuple<vector<double>, vector<double>> &tuple_vect, 
                        const tuple<vector<double>, vector<double>> &tuple_mean, 
                        vector<double> &arr_out, size_t &w, int num_threads){

    auto& vect1 = std::get<0>(tuple_vect);
    auto& vect2 = std::get<1>(tuple_vect);
    int max_length = min(vect1.size(), vect2.size());
    int chunk = max(1, int(max_length/num_threads));
    pthread_t th[num_threads];

    // Creation of the sing thread function arguments
    ThreadArgs_corr args[num_threads]; 

    cout << "s f s g \n";
    // Each thread has its own arguments
    for(int jj=0; jj<num_threads;jj++){
        args[jj].tuple_vect  = &tuple_vect;
        args[jj].tuple_mean = &tuple_mean;
        args[jj].arr_out = &arr_out;
        args[jj].w       = w;
        
        // Range division
        args[jj].start_index = jj * chunk;
        args[jj].end_index   = (jj == num_threads - 1) ? max_length : (jj + 1) * chunk+w;        

        pthread_create(&th[jj], NULL, &rolling_corr_single_thr_exe, &args[jj]);
    }

    // wait for threads to finish
    for (int jj = 0; jj < num_threads; jj++) {
        pthread_join(th[jj], NULL);
    }

    return;
}









void* rolling_mean_parallel_interface(void* arg_inp){

    ThreadArgs_mean* state = static_cast<ThreadArgs_mean*>(arg_inp);

    int num_threads = state->num_threads;
    int chunk = (*state->arr_in).size()/(num_threads);

    pthread_t th[num_threads];

    // Creation of the sing thread function arguments
    ThreadArgs_mean args[num_threads]; 

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

    ThreadArgs_mean* state = static_cast<ThreadArgs_mean*>(arg_inp);

    rolling_mean_exec(*state->arr_in, *state->arr_out, state->w);
    
    return nullptr;
}

void rolling_mean_parallel_inputs(vector<vector<double>> &arrs_in, vector<vector<double>> &arrs_out,
                                     size_t &w, int num_threads, bool nested_threads){


    pthread_t th[arrs_in.size()];

    // Creation of the sing thread function arguments
    ThreadArgs_mean args[arrs_in.size()]; 

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