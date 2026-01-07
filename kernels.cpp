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

void rolling_mean_corr_exec_mv(const vector<vector<double>> &vect, 
                        vector<vector<double>> &vect_mean, 
                        vector<vector<vector<double>>> &arr_out, size_t &w, int start_index, int end_index){

    int N_vect  = vect.size();
    int n_ele   = vect[0].size();
    for (int j=0; j<N_vect; ++j) vect_mean[j][start_index] = 0.0;

    if (end_index == -1){
        end_index = (int)vect[0].size();
        for (int ii=1;ii<vect.size();ii++){end_index = min((int)vect[ii].size(), end_index);};
    }
        
    double denom, dummy_mean;

    vector<vector<double>> Svv(N_vect, vector<double>(N_vect)), cov_vv(N_vect, vector<double>(N_vect));  

    for (int ii = start_index; ii < start_index+w; ii++){
        for (int jj=0; jj<Svv.size();jj++){
            for(int kk=jj; kk<Svv.size();kk++){
                Svv[jj][kk] += vect[jj][ii] * vect[kk][ii];
            }
           vect_mean[jj][start_index] += vect[jj][ii]/w;
        }
    }
    for (int jj=0; jj<Svv.size();jj++){ 
        for(int kk=jj; kk<Svv.size();kk++){
            cov_vv[jj][kk] = Svv[jj][kk]/w - vect_mean[jj][start_index] *vect_mean[kk][start_index];
        }
    }

    for (int jj=0; jj<Svv.size();jj++){ 
        arr_out[jj][jj][start_index] = 1;
        for(int kk=jj+1; kk<Svv.size();kk++){
            denom = std::sqrt(cov_vv[jj][jj] * cov_vv[kk][kk]);
            arr_out[jj][kk][start_index] = (denom > 0) ? cov_vv[jj][kk]/denom : 0.0;
            arr_out[kk][jj][start_index] = arr_out[jj][kk][start_index];
        }
    }

    // DP part for the rolling window
    for (int ii = start_index+1; ii < end_index-w; ii++){

        for (int jj=0; jj<Svv.size();jj++){
            for(int kk=jj; kk<Svv.size();kk++){
                Svv[jj][kk] += vect[jj][ii+w-1] * vect[kk][ii+w-1] - vect[jj][ii-1] * vect[kk][ii-1];
                dummy_mean   = (w*vect_mean[kk][ii-1] - vect[kk][ii-1] + vect[kk][ii+w-1])/w;
                cov_vv[jj][kk] = Svv[jj][kk]/w - vect_mean[jj][ii] * dummy_mean;
            }
        }
        for (int jj=0; jj<Svv.size();jj++){ 
            arr_out[jj][jj][ii] = 1.0;
            for(int kk=jj+1; kk<Svv.size();kk++){
                denom = std::sqrt(cov_vv[jj][jj] * cov_vv[kk][kk]);
                arr_out[jj][kk][ii] = (denom > 0) ? cov_vv[jj][kk]/denom : 0.0;
                arr_out[kk][jj][ii] = arr_out[jj][kk][ii];
                
                if (abs(arr_out[kk][jj][ii])>1 ) printf("error in the algorithm, correlation larger than 1.\n");

            }
        }
    }
    return;
}



void rolling_mean_corr_exec(const vector<vector<double>> &vect, 
                        vector<vector<double>> &vect_mean, 
                        vector<double> &arr_out, size_t &w, int start_index, int end_index){

    auto& vect1 = vect[0];
    auto& vect2 = vect[1];
    auto& vect1_mean = vect_mean[0];
    auto& vect2_mean = vect_mean[1];

    if (end_index == -1) end_index = min(vect1.size(), vect2.size());
    
    double cov12 = 0.0;
    double std1  = 0.0;
    double std2  = 0.0;
    double denom;
    // 1. compute the first corr
    double Sv1v1(0.0), Sv2v2(0.0), Sv1v2(0.0);
    double mean_v1(0.0), mean_v2(0.0);
    for (int ii = start_index; ii < start_index+w; ii++){
        Sv1v1 += vect1[ii]* vect1[ii];
        Sv1v2 += vect1[ii]* vect2[ii];
        Sv2v2 += vect2[ii]* vect2[ii];
        mean_v1 += vect1[ii];
        mean_v2 += vect2[ii];
    }
    vect1_mean[start_index] = mean_v1/w;
    vect2_mean[start_index] = mean_v2/w;

    std1    = Sv1v1/w - vect1_mean[start_index] * vect1_mean[start_index];
    std2    = Sv2v2/w - vect2_mean[start_index] * vect2_mean[start_index];
    cov12 = Sv1v2/w - vect1_mean[start_index] * vect2_mean[start_index];

    denom = std::sqrt(std1*std2);
    arr_out[start_index] = (denom > 0) ? cov12/denom : 0.0;

    // DP part for the rolling window
    for (int ii = start_index+1; ii < end_index-w; ii++){
        Sv1v1 += vect1[ii+w-1]* vect1[ii+w-1] - vect1[ii-1]* vect1[ii-1] ;
        Sv1v2 += vect1[ii+w-1]* vect2[ii+w-1] - vect1[ii-1]* vect2[ii-1] ;
        Sv2v2 += vect2[ii+w-1]* vect2[ii+w-1] - vect2[ii-1]* vect2[ii-1] ;

        vect1_mean[ii] = (w*vect1_mean[ii-1] - vect1[ii-1] + vect1[ii+w-1])/w;
        vect2_mean[ii] = (w*vect2_mean[ii-1] - vect2[ii-1] + vect2[ii+w-1])/w;
    
        std1    = Sv1v1/w - vect1_mean[ii] * vect1_mean[ii];
        std2    = Sv2v2/w - vect2_mean[ii] * vect2_mean[ii];
        cov12   = Sv1v2/w - vect1_mean[ii] * vect2_mean[ii];

        denom = std::sqrt(std1*std2);
        arr_out[ii] = (denom > 0) ? cov12/denom : 0.0;

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
struct Thread_Args {
    const vector<double>* arr_in;
    vector<vector<vector<double>>>* arr_out;
    const vector<vector<double>>* vect;
    vector<vector<double>>* vect_mean;
    vector<double>* arr_out_mean;
    size_t w;
    int start_index;
    int end_index;
    int num_threads;
    string method;
};



// single thread execution function
void* single_thr_exe_interface(void* arg){

    // Cast the void pointer to its structure
    Thread_Args* state = static_cast<Thread_Args*>(arg);
    if (state->method == "mean")         rolling_mean_exec(*state->arr_in, *state->arr_out_mean, state->w, state->start_index, state->end_index);
    if (state->method == "correlation")  rolling_mean_corr_exec_mv(*state->vect, *state->vect_mean, *state->arr_out, state->w, state->start_index, state->end_index);

    return nullptr; 
}



void rolling_mean_parallel(vector<double> &arr_in, vector<double> &arr_out, size_t &w, int num_threads){

    string method = "mean";

    int chunk = arr_in.size()/num_threads;

    pthread_t th[num_threads];

    // Creation of the sing thread function arguments
    Thread_Args args[num_threads]; 

    // Each thread has its own arguments
    for(int jj=0; jj<num_threads;jj++){
        args[jj].arr_in  = &arr_in;
        // args[jj].arr_out = &arr_out;
        args[jj].arr_out_mean = &arr_out;
        args[jj].w       = w;
        args[jj].method  = method;

        // Range division
        args[jj].start_index = jj * chunk;
        args[jj].end_index   = (jj == num_threads - 1) ? arr_in.size() : (jj + 1) * chunk+w;        

        pthread_create(&th[jj], NULL, &single_thr_exe_interface, &args[jj]);
    }

    // wait for threads to finish
    for (int jj = 0; jj < num_threads; jj++) {
        pthread_join(th[jj], NULL);
    }

    return;
}

void rolling_corr_parallel(const std::vector<vector<double>> &vect,
                           vector<vector<double>> &vect_mean,
                           vector<vector<vector<double>>> &arr_out,
                           size_t &w, int num_threads) {

    if (vect.size() < 2)  throw std::runtime_error("rolling_corr_parallel: need at least 2 vectors");
    if (num_threads <= 0) throw std::runtime_error("rolling_corr_parallel: num_threads must be > 0");

    std::string method = "correlation";
    auto& vect1 = vect[0];
    auto& vect2 = vect[1];

    int max_length = (int)std::min(vect1.size(), vect2.size());
    int chunk = std::max(1, (max_length / num_threads));

    pthread_t th[num_threads];
    Thread_Args args[num_threads];

    for (int jj = 0; jj < num_threads; jj++) {
        args[jj].vect       = &vect;        // <--- changed
        args[jj].vect_mean  = &vect_mean;
        args[jj].arr_out    = &arr_out;
        args[jj].w          = w;
        args[jj].method     = method;

        args[jj].start_index = jj * chunk;
        args[jj].end_index   = (jj == num_threads - 1) ? max_length : (jj + 1) * chunk + (int)w;

        pthread_create(&th[jj], NULL, &single_thr_exe_interface, &args[jj]);
    }

    for (int jj = 0; jj < num_threads; jj++) {
        pthread_join(th[jj], NULL);
    }
}









void* rolling_mean_parallel_interface(void* arg_inp){

    Thread_Args* state = static_cast<Thread_Args*>(arg_inp);

    int num_threads = state->num_threads;
    int chunk = (*state->arr_in).size()/(num_threads);

    pthread_t th[num_threads];

    // Creation of the sing thread function arguments
    Thread_Args args[num_threads]; 

    // Each thread has its own arguments
    for(int jj=0; jj<num_threads;jj++){
        args[jj].arr_in  = state->arr_in;
        args[jj].arr_out = state->arr_out;
        args[jj].w       = state->w;
        args[jj].num_threads  = num_threads;
        
        // Range division
        args[jj].start_index = jj * chunk;
        args[jj].end_index = (jj == num_threads - 1) ? (*state->arr_in).size() : (jj + 1) * chunk+state->w;        

        pthread_create(&th[jj], NULL, &single_thr_exe_interface, &args[jj]);
    }

    // wait for threads to finish
    for (int jj = 0; jj < num_threads; jj++) {
        pthread_join(th[jj], NULL);
    }

    return nullptr;
}



void* rolling_mean_exe_interface(void* arg_inp){

    Thread_Args* state = static_cast<Thread_Args*>(arg_inp);

    rolling_mean_exec(*state->arr_in, *state->arr_out_mean, state->w);
    
    return nullptr;
}

void rolling_mean_parallel_inputs(vector<vector<double>> &arrs_in, vector<vector<double>> &arrs_out,
                                     size_t &w, int num_threads, bool nested_threads){


    pthread_t th[arrs_in.size()];

    // Creation of the sing thread function arguments
    Thread_Args args[arrs_in.size()]; 

    // Each thread has its own arguments
    // for(int jj=0; jj<arrs_in.size();jj++){
    for(int jj=0; jj<arrs_in.size();jj++){
        args[jj].arr_in  = &arrs_in[jj];
        args[jj].arr_out_mean = &arrs_out[jj];
        args[jj].w       = w;
        args[jj].num_threads = std::max(1,int(num_threads/arrs_in.size()));
        
        // Range division
        args[jj].start_index = 0;
        args[jj].end_index   = arrs_in[jj].size();        

        if(nested_threads){
            if (jj==0) printf("using a total of %d threads for par vect input and par vect treat \n \n", args[jj].num_threads * arrs_in.size());
            pthread_create(&th[jj], NULL, &rolling_mean_parallel_interface, &args[jj]);
        }else{         
            pthread_create(&th[jj], NULL, &rolling_mean_exe_interface, &args[jj]);
        }
    }

    // wait for threads to finish
    for (int jj = 0; jj < arrs_in.size(); jj++) {
        pthread_join(th[jj], NULL);
    }

    return;
}


}