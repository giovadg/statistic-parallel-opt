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
                        vector<vector<double>> &vect_var,                         
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

        vect_var[jj][start_index] = Svv[jj][jj] - vect_mean[jj][start_index]*vect_mean[jj][start_index];

        for(int kk=jj; kk<Svv.size();kk++){
            cov_vv[jj][kk] = Svv[jj][kk]/w - vect_mean[jj][start_index] *vect_mean[kk][start_index];
        }
    }

    for (int jj=0; jj<Svv.size();jj++){ 
        arr_out[jj][jj][start_index] = 1;
        for(int kk=jj+1; kk<Svv.size();kk++){
            denom = sqrt(cov_vv[jj][jj] * cov_vv[kk][kk]);
            arr_out[jj][kk][start_index] = (denom > 0) ? cov_vv[jj][kk]/denom : 0.0;
            arr_out[kk][jj][start_index] = arr_out[jj][kk][start_index];
            if (abs(arr_out[kk][jj][start_index])>1 ) {
                printf("errore larger than 1.\n");
                cout<< cov_vv[jj][kk]<< " "<< cov_vv[jj][jj] << " "<< cov_vv[kk][kk]<< " "<< jj<< 
                        " "<< kk<<" "<<vect[jj][start_index]<< " "<<vect[kk][start_index]<<endl;
            }
        }
    }

    // DP part for the rolling window
    for (int ii = start_index+1; ii <= end_index-w; ii++){
        for (int jj=0; jj<Svv.size();jj++){
            vect_mean[jj][ii] = (w*vect_mean[jj][ii-1] - vect[jj][ii-1] + vect[jj][ii+w-1])/w;
            for(int kk=jj; kk<Svv.size();kk++){
                Svv[jj][kk] += vect[jj][ii+w-1] * vect[kk][ii+w-1] - vect[jj][ii-1] * vect[kk][ii-1];
                dummy_mean   = (w*vect_mean[kk][ii-1] - vect[kk][ii-1] + vect[kk][ii+w-1])/w;
                cov_vv[jj][kk] = Svv[jj][kk]/w - vect_mean[jj][ii] * dummy_mean;
            }
            vect_var[jj][ii] = Svv[jj][jj] - vect_mean[jj][ii]*vect_mean[jj][ii];
        }

        for (int jj=0; jj<Svv.size();jj++){ 
            arr_out[jj][jj][ii] = 1.0;
            for(int kk=jj+1; kk<Svv.size();kk++){
                denom = sqrt(cov_vv[jj][jj] * cov_vv[kk][kk]);
                arr_out[jj][kk][ii] = (denom > 1e-11) ? cov_vv[jj][kk]/denom : 0.0;
                arr_out[kk][jj][ii] = arr_out[jj][kk][ii];
                
                if (abs(arr_out[kk][jj][ii])>1 ) {
                    printf("error in the algorithm, correlation larger than 1.\n");
                    cout<< cov_vv[jj][kk]<< " "<< cov_vv[jj][jj] << " "<< cov_vv[kk][kk]<< " "<< jj<< 
                            " "<< kk<<" "<<vect[jj][ii+w-1]<< " "<<vect[kk][ii+w-1]<<endl;
                }

            }
        }
    }
    return;
}




void rolling_var_exec(const vector<vector<double>> &arr_in, vector<vector<double>> &arr_mean, 
                        vector<vector<double>> &arr_var,
                         size_t &w, int start_index, int end_index, int vect_start, int vect_end){

    if (end_index == -1) end_index = (int)arr_in[0].size();
    if (vect_end  == -1) vect_end  = (int)arr_in.size();

    auto lamb = [] (double aa) {return aa*aa;};
    double sum_sq, sum;
    // 1. compute the first var
    for (int jj=vect_start; jj<vect_end;jj++){
        sum_sq = std::transform_reduce(arr_in[jj].begin() + start_index,arr_in[jj].begin() + start_index + w, 
                                                0.0, std::plus<>(), lamb);

        sum    = std::accumulate(arr_in[jj].begin()+start_index, arr_in[jj].begin()+start_index+w, 0.0);

        arr_var[jj][start_index]    = sum_sq/w - (sum/w)*(sum/w);
        arr_mean[jj][start_index]   = sum/w;
    }    

    // 3. Computes the other averages using DP
    //    Iterate from the starting index + 1 until the last available point   
    for (int jj=vect_start; jj<vect_end;jj++){
        for (int ii=start_index+1; ii<=end_index-w;ii++){
            sum_sq = sum_sq - lamb(arr_in[jj][ii-1]) + lamb(arr_in[jj][ii+w-1]);

            arr_mean[jj][ii] = (w*arr_mean[jj][ii-1] - arr_in[jj][ii-1] + arr_in[jj][ii+w-1])/w;

            arr_var[jj][ii]  = sum_sq/w - arr_mean[jj][ii]*arr_mean[jj][ii];

        }
    }

    return;
}


void rolling_mean_exec(const vector<vector<double>> &arr_in, vector<vector<double>> &arr_out,
                         size_t &w, int start_index, int end_index, int vect_start, int vect_end){

    if (end_index == -1) end_index = (int)arr_in[0].size();
    if (vect_end  == -1) vect_end  = (int)arr_in.size();
    double sum;
    // 1. compute the first average
    for (int jj=vect_start; jj<vect_end;jj++){
        sum = std::accumulate(arr_in[jj].begin()+start_index, arr_in[jj].begin()+start_index+w, 0.0);
        arr_out[jj][start_index] = sum/w;
    }
    // 3. Computes the other averages using DP
    //    Iterate from the starting index + 1 until the last available point   
    for (int ii=start_index+1; ii<=end_index-w;ii++){
        for (int jj=vect_start; jj<vect_end;jj++){
            arr_out[jj][ii] = (w*arr_out[jj][ii-1] - arr_in[jj][ii-1] + arr_in[jj][ii+w-1])/w;
        }
    }

    return;
}

// define the structure of each thread input
struct Thread_Args {
    const vector<vector<double>>* vect;
    vector<vector<vector<double>>>* arr_out;
    vector<vector<double>>* vect_mean;
    vector<vector<double>>* vect_var;
    vector<double>* arr_out_mean;
    size_t w;
    int start_index;
    int end_index;
    int num_threads;
    int vect_start;
    int vect_end;
    string method;
};



// single thread execution function
void* single_thr_exe_interface(void* arg){

    // Cast the void pointer to its structure
    Thread_Args* state = static_cast<Thread_Args*>(arg);
    if (state->method == "mean")         rolling_mean_exec(*state->vect, *state->vect_mean, state->w, state->start_index, state->end_index,
                                                        state->vect_start, state->vect_end );

    if (state->method == "variance")     rolling_var_exec(*state->vect, *state->vect_mean, *state->vect_var, state->w, state->start_index, state->end_index,
                                                        state->vect_start, state->vect_end );

    if (state->method == "correlation")  rolling_mean_corr_exec_mv(*state->vect, *state->vect_mean, *state->vect_var, *state->arr_out, state->w, state->start_index, state->end_index);

    return nullptr; 
}



void rolling_stat_parallel(const vector<vector<double>> &arr_in, vector<vector<double>> &arr_mean,
                         vector<vector<double>> &arr_var, string method, size_t &w, int num_threads){

    int chunk = (int)arr_in[0].size()/num_threads;

    pthread_t th[num_threads];

    // Creation of the sing thread function arguments
    Thread_Args args[num_threads]; 

    // Each thread has its own arguments
    for(int jj=0; jj<num_threads;jj++){
        args[jj].vect       = &arr_in;
        args[jj].vect_mean  = &arr_mean;
        args[jj].vect_var   = &arr_var;
        args[jj].w       = w;
        args[jj].method  = method;
        // Range vectors
        args[jj].vect_start  = 0;
        args[jj].vect_end    = (int)arr_in.size();  
        // Range division
        args[jj].start_index = jj * chunk;
        args[jj].end_index   = (jj == num_threads - 1) ? (int)arr_in[0].size() : (jj + 1) * chunk+w;        

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
                           vector<vector<double>> &vect_var,
                           vector<vector<vector<double>>> &arr_out,
                           size_t &w, int num_threads) {

    if (vect.size() < 2)  throw std::runtime_error("rolling_corr_parallel: need at least 2 vectors");
    if (num_threads <= 0) throw std::runtime_error("rolling_corr_parallel: num_threads must be > 0");

    string method = "correlation";


    int max_length = (int)vect[0].size();
    for (int ii=1;ii<vect.size();ii++){max_length = min(max_length, (int)vect[ii].size());}

    int threads_to_use = std::min(num_threads, max_length); // evita chunk=0 e range degeneri
    // int chunk = max(1, (max_length / num_threads));
    int chunk = (max_length + threads_to_use - 1) / threads_to_use; // ceil

    pthread_t th[threads_to_use];
    Thread_Args args[threads_to_use];

    for (int jj = 0; jj < threads_to_use; jj++) {
        args[jj].vect       = &vect;        // <--- changed
        args[jj].vect_mean  = &vect_mean;
        args[jj].vect_var   = &vect_var;
        args[jj].arr_out    = &arr_out;
        args[jj].w          = w;
        args[jj].method     = method;

        args[jj].start_index = jj * chunk;
        args[jj].end_index   = min((jj + 1) * chunk + (int)w, max_length);

        pthread_create(&th[jj], NULL, &single_thr_exe_interface, &args[jj]);
    }

    for (int jj = 0; jj < threads_to_use; jj++) {
        pthread_join(th[jj], NULL);
    }
}









void* rolling_stat_parallel_interface(void* arg_inp){

    Thread_Args* state = static_cast<Thread_Args*>(arg_inp);

    int num_threads = state->num_threads;
    int chunk = ((int)(*state->vect)[0].size())/(num_threads);

    pthread_t th[num_threads];

    // Creation of the sing thread function arguments
    Thread_Args args[num_threads]; 

    // Each thread has its own arguments
    for(int jj=0; jj<num_threads;jj++){
        args[jj].vect         = state->vect;
        args[jj].vect_mean    = state->vect_mean;
        args[jj].vect_var    = state->vect_var;
        args[jj].w            = state->w;
        args[jj].num_threads  = num_threads;
        args[jj].vect_start   = state->vect_start;
        args[jj].vect_end     = state->vect_end;
        args[jj].method       = state->method;

        // Range division
        args[jj].start_index = jj * chunk;
        args[jj].end_index = (jj == num_threads - 1) ? (*state->vect)[0].size() : (jj + 1) * chunk+state->w;        

        pthread_create(&th[jj], NULL, &single_thr_exe_interface, &args[jj]);
    }

    // wait for threads to finish
    for (int jj = 0; jj < num_threads; jj++) {
        pthread_join(th[jj], NULL);
    }

    return nullptr;
}





void rolling_stat_parallel_nested(const vector<vector<double>> &arrs_in, vector<vector<double>> &arrs_mean, 
                                     vector<vector<double>> &arrs_var, 
                                     string method, size_t &w, int num_threads, bool nested_threads){

    pthread_t th[arrs_in.size()];

    // Creation of the sing thread function arguments
    Thread_Args args[arrs_in.size()]; 

    int N_vect       = arrs_in.size();

    int N_vect_chunk = std::max(1,(int)arrs_in.size()/num_threads);

    // Each thread has its own arguments
    for(int jj=0; jj<arrs_in.size();jj=jj+N_vect_chunk){
        args[jj].vect      = &arrs_in;
        args[jj].vect_mean = &arrs_mean;
        args[jj].vect_var  = &arrs_var;
        args[jj].w         = w;
        args[jj].num_threads = std::max(1,int(num_threads/arrs_in.size()));
        args[jj].vect_start  = jj;
        args[jj].vect_end = (jj+N_vect_chunk < N_vect) ? jj+N_vect_chunk : N_vect;
        args[jj].method   = method;

        // Range division
        args[jj].start_index = 0;
        args[jj].end_index   = arrs_in[jj].size();        


        if(nested_threads){
            if (jj==0) printf("using a total of %d threads for par vect input and par vect treat \n \n", args[jj].num_threads * arrs_in.size());
            pthread_create(&th[jj], NULL, &rolling_stat_parallel_interface, &args[jj]);
        }else{         
            pthread_create(&th[jj], NULL, &single_thr_exe_interface, &args[jj]);
        }
    }

    // wait for threads to finish
    for (int jj = 0; jj < arrs_in.size(); jj++) {
        pthread_join(th[jj], NULL);
    }

    return;
}


}