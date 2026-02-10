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


inline size_t idx3(size_t j, size_t k, size_t t,
                   size_t N, size_t T) {
    return (j*N + k)*T + t;   // n_ele continuous in memory
}

namespace kernels {

// Numerically stable rolling correlation.
// Implements a Welford-style remove/add update on centered residuals,
// avoiding catastrophic cancellation present in E[x^2] - E[x]^2 formulations.
// Computes rolling covariance and correlation in O(1) per step.  
void rolling_mean_corr_exec_mv_1Dout(const vector<vector<double>> &vect, 
                        vector<vector<double>> &vect_mean, 
                        vector<vector<double>> &vect_var,                         
                        vector<double> &arr_out, size_t &w, int start_index, int end_index){

    int N_vect  = vect.size();
    int n_ele   = vect[0].size();
    double inv_w = 1/(double)w;
    for (int j=0; j<N_vect; ++j) vect_mean[j][start_index] = 0.0;

    if (end_index == -1){
        end_index = (int)vect[0].size();
        for (int ii=1;ii<vect.size();ii++){end_index = min((int)vect[ii].size(), end_index);};
    }
        
    double denom;

    vector<vector<double>> Svv(N_vect, vector<double>(N_vect)), cov_vv(N_vect, vector<double>(N_vect));  
    vector<double> mu_mid(N_vect); 

    for (int ii = start_index; ii < start_index+w; ii++){
        for (int jj=0; jj<Svv.size();jj++){
            for(int kk=jj; kk<Svv.size();kk++){
                Svv[jj][kk] += vect[jj][ii] * vect[kk][ii];
            }
           vect_mean[jj][start_index] += vect[jj][ii]/w;
        }
    }
    for (int jj=0; jj<Svv.size();jj++){ 

        vect_var[jj][start_index] = Svv[jj][jj]/w - vect_mean[jj][start_index]*vect_mean[jj][start_index];

        for(int kk=jj; kk<Svv.size();kk++){
            cov_vv[jj][kk] = Svv[jj][kk]/w - vect_mean[jj][start_index] *vect_mean[kk][start_index];
        }
    }

    for (int jj=0; jj<Svv.size();jj++){ 
        arr_out[idx3(jj,jj,start_index,N_vect,n_ele)] = 1;
        for(int kk=jj+1; kk<Svv.size();kk++){
            denom = sqrt(vect_var[jj][start_index]  * vect_var[kk][start_index] );
            
            // arr_out[jj][kk][start_index] = (denom > 0) ? cov_vv[jj][kk]/denom : 0.0;
            arr_out[idx3(jj,kk,start_index,N_vect,n_ele)] = (denom > 0) ? cov_vv[jj][kk]/denom : 0.0;
            arr_out[idx3(kk,jj,start_index,N_vect,n_ele)] = arr_out[idx3(jj,kk,start_index,N_vect,n_ele)];
            if (abs(arr_out[idx3(kk,jj,start_index,N_vect,n_ele)])>1 ) {
                printf("errore larger than 1.\n");
                cout<< cov_vv[jj][kk]<< " "<< vect_var[jj][start_index] << " "<< vect_var[kk][start_index]<< " "<< jj<< " "<< kk<<endl;
            }
        }
    }

    // DP part for the rolling window
    for (int ii = start_index+1; ii <= end_index-w; ii++){
        double C_mid(0);
        {
            
            int jj=0;
            arr_out[idx3(jj,jj,ii,N_vect,n_ele)] = 1;
            vect_mean[jj][ii] =  vect_mean[jj][ii-1] + inv_w * (vect[jj][ii+w-1] - vect[jj][ii-1]);

            mu_mid[jj]        = (w* vect_mean[jj][ii-1] - vect[jj][ii-1])/(w - 1);

            C_mid            = w*cov_vv[jj][jj] - (vect[jj][ii-1] - vect_mean[jj][ii-1]) * (vect[jj][ii-1] - mu_mid[jj]);
            cov_vv[jj][jj]   = inv_w * (C_mid    + (vect[jj][ii+w-1]  - mu_mid[jj]) * (vect[jj][ii+w-1]  - vect_mean[jj][ii]));
            vect_var[jj][ii] = cov_vv[jj][jj];
            
            for(int kk=jj+1; kk<cov_vv.size();kk++){

                vect_mean[kk][ii] = vect_mean[kk][ii-1] + inv_w * (vect[kk][ii+w-1] - vect[kk][ii-1]);

                mu_mid[kk]        = (w* vect_mean[kk][ii-1] - vect[kk][ii-1])/(w - 1);

                C_mid            = w * cov_vv[kk][kk] - (vect[kk][ii-1] - vect_mean[kk][ii-1]) * (vect[kk][ii-1] - mu_mid[kk]);
                cov_vv[kk][kk]   = inv_w * (C_mid    + (vect[kk][ii+w-1]  - mu_mid[kk]) * (vect[kk][ii+w-1]  - vect_mean[kk][ii]));
                vect_var[kk][ii] = cov_vv[kk][kk];

                C_mid             = w*cov_vv[jj][kk] - (vect[jj][ii-1] - vect_mean[jj][ii-1]) * (vect[kk][ii-1] - mu_mid[kk]);
                cov_vv[jj][kk]    = inv_w * (C_mid + (vect[jj][ii+w-1] - mu_mid[jj]) * (vect[kk][ii+w-1] - vect_mean[kk][ii]));

                denom = sqrt(vect_var[jj][ii] * vect_var[kk][ii]);

                arr_out[idx3(kk,jj,ii,N_vect,n_ele)] = (denom > 1e-11) ? cov_vv[jj][kk]/denom : 0.0;
                arr_out[idx3(jj,kk,ii,N_vect,n_ele)] = arr_out[idx3(kk,jj,ii,N_vect,n_ele)];
            }
        } ;

        for (int jj=1; jj<Svv.size();jj++){
            arr_out[idx3(jj,jj,ii,N_vect,n_ele)] = 1.0;
            for(int kk=jj+1; kk<Svv.size();kk++){

                C_mid             = w*cov_vv[jj][kk] - (vect[jj][ii-1] - vect_mean[jj][ii-1]) * (vect[kk][ii-1] - mu_mid[kk]);

                cov_vv[jj][kk]    = inv_w * (C_mid + (vect[jj][ii+w-1] - mu_mid[jj]) * (vect[kk][ii+w-1] - vect_mean[kk][ii]));
 
                denom = sqrt(vect_var[jj][ii] * vect_var[kk][ii]);

                arr_out[idx3(kk,jj,ii,N_vect,n_ele)] = (denom > 1e-11) ? cov_vv[jj][kk]/denom : 0.0;
                arr_out[idx3(jj,kk,ii,N_vect,n_ele)] = arr_out[idx3(kk,jj,ii,N_vect,n_ele)];

                if (abs(arr_out[idx3(jj,kk,ii,N_vect,n_ele)])>1 ) {
                    printf("error in the algorithm, correlation larger than 1.\n");
                    cout<< cov_vv[jj][kk]<< " "<< vect_var[jj][ii] << " "<< vect_var[kk][ii]<< " "<< jj<< " "<< kk<<endl;
                }
            }
        }
    }
    return;
}


// Numerically stable rolling correlation.
// Implements a Welford-style remove/add update on centered residuals,
// avoiding catastrophic cancellation present in E[x^2] - E[x]^2 formulations.
// Computes rolling covariance and correlation in O(1) per step.  
void rolling_mean_corr_exec_mv(const vector<vector<double>> &vect, 
                        vector<vector<double>> &vect_mean, 
                        vector<vector<double>> &vect_var,                         
                        vector<vector<vector<double>>> &arr_out, size_t &w, int start_index, int end_index){

    int N_vect  = vect.size();
    int n_ele   = vect[0].size();
    double inv_w = 1/(double)w;
    for (int j=0; j<N_vect; ++j) vect_mean[j][start_index] = 0.0;

    if (end_index == -1){
        end_index = (int)vect[0].size();
        for (int ii=1;ii<vect.size();ii++){end_index = min((int)vect[ii].size(), end_index);};
    }
        
    double denom;

    vector<vector<double>> Svv(N_vect, vector<double>(N_vect)), cov_vv(N_vect, vector<double>(N_vect));  
    vector<double> mu_mid(N_vect); 

    for (int ii = start_index; ii < start_index+w; ii++){
        for (int jj=0; jj<Svv.size();jj++){
            for(int kk=jj; kk<Svv.size();kk++){
                Svv[jj][kk] += vect[jj][ii] * vect[kk][ii];
            }
           vect_mean[jj][start_index] += vect[jj][ii]/w;
        }
    }
    for (int jj=0; jj<Svv.size();jj++){ 

        vect_var[jj][start_index] = Svv[jj][jj]/w - vect_mean[jj][start_index]*vect_mean[jj][start_index];

        for(int kk=jj; kk<Svv.size();kk++){
            cov_vv[jj][kk] = Svv[jj][kk]/w - vect_mean[jj][start_index] *vect_mean[kk][start_index];
        }
    }

    for (int jj=0; jj<Svv.size();jj++){ 
        arr_out[jj][jj][start_index] = 1;
        for(int kk=jj+1; kk<Svv.size();kk++){
            denom = sqrt(vect_var[jj][start_index]  * vect_var[kk][start_index] );
            
            arr_out[jj][kk][start_index] = (denom > 0) ? cov_vv[jj][kk]/denom : 0.0;
            arr_out[kk][jj][start_index] = arr_out[jj][kk][start_index];
            if (abs(arr_out[kk][jj][start_index])>1 ) {
                printf("errore larger than 1.\n");
                cout<< cov_vv[jj][kk]<< " "<< vect_var[jj][start_index] << " "<< vect_var[kk][start_index]<< " "<< jj<< " "<< kk<<endl;
            }
        }
    }

    // DP part for the rolling window
    for (int ii = start_index+1; ii <= end_index-w; ii++){
        double C_mid(0);
        {
            
            int jj=0;
            arr_out[jj][jj][ii] = 1.0;

            vect_mean[jj][ii] =  vect_mean[jj][ii-1] + inv_w * (vect[jj][ii+w-1] - vect[jj][ii-1]);

            mu_mid[jj]        = (w* vect_mean[jj][ii-1] - vect[jj][ii-1])/(w - 1);

            C_mid            = w*cov_vv[jj][jj] - (vect[jj][ii-1] - vect_mean[jj][ii-1]) * (vect[jj][ii-1] - mu_mid[jj]);
            cov_vv[jj][jj]   = inv_w * (C_mid    + (vect[jj][ii+w-1]  - mu_mid[jj]) * (vect[jj][ii+w-1]  - vect_mean[jj][ii]));
            vect_var[jj][ii] = cov_vv[jj][jj];
            
            for(int kk=jj+1; kk<cov_vv.size();kk++){

                vect_mean[kk][ii] = vect_mean[kk][ii-1] + inv_w * (vect[kk][ii+w-1] - vect[kk][ii-1]);

                mu_mid[kk]        = (w* vect_mean[kk][ii-1] - vect[kk][ii-1])/(w - 1);

                C_mid            = w * cov_vv[kk][kk] - (vect[kk][ii-1] - vect_mean[kk][ii-1]) * (vect[kk][ii-1] - mu_mid[kk]);
                cov_vv[kk][kk]   = inv_w * (C_mid    + (vect[kk][ii+w-1]  - mu_mid[kk]) * (vect[kk][ii+w-1]  - vect_mean[kk][ii]));
                vect_var[kk][ii] = cov_vv[kk][kk];

                C_mid             = w*cov_vv[jj][kk] - (vect[jj][ii-1] - vect_mean[jj][ii-1]) * (vect[kk][ii-1] - mu_mid[kk]);
                cov_vv[jj][kk]    = inv_w * (C_mid + (vect[jj][ii+w-1] - mu_mid[jj]) * (vect[kk][ii+w-1] - vect_mean[kk][ii]));

                denom = sqrt(vect_var[jj][ii] * vect_var[kk][ii]);

                arr_out[jj][kk][ii] = (denom > 1e-11) ? cov_vv[jj][kk]/denom : 0.0;
                arr_out[kk][jj][ii] = arr_out[jj][kk][ii];
            }
        } ;

        for (int jj=1; jj<Svv.size();jj++){
            arr_out[jj][jj][ii] = 1.0;
            for(int kk=jj+1; kk<Svv.size();kk++){

                C_mid             = w*cov_vv[jj][kk] - (vect[jj][ii-1] - vect_mean[jj][ii-1]) * (vect[kk][ii-1] - mu_mid[kk]);

                cov_vv[jj][kk]    = inv_w * (C_mid + (vect[jj][ii+w-1] - mu_mid[jj]) * (vect[kk][ii+w-1] - vect_mean[kk][ii]));
 
                denom = sqrt(vect_var[jj][ii] * vect_var[kk][ii]);

                arr_out[jj][kk][ii] = (denom > 1e-11) ? cov_vv[jj][kk]/denom : 0.0;
                arr_out[kk][jj][ii] = arr_out[jj][kk][ii];

                if (abs(arr_out[kk][jj][ii])>1 ) {
                    printf("error in the algorithm, correlation larger than 1.\n");
                    cout<< cov_vv[jj][kk]<< " "<< vect_var[jj][ii] << " "<< vect_var[kk][ii]<< " "<< jj<< " "<< kk<<endl;
                }
            }
        }
    }
    return;
}


// Naive function for computing the corring window correlation:
// it uses the dynamic programming (DP) on the x_i y_i moment. 
// Can suffer from cancelletion problem  
void rolling_mean_corr_exec_mv_simple(const vector<vector<double>> &vect, 
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
        
    double denom;

    vector<vector<double>> Svv(N_vect, vector<double>(N_vect)), cov_vv(N_vect, vector<double>(N_vect));  
    vector<double> mu_mid(N_vect); 

    for (int ii = start_index; ii < start_index+w; ii++){
        for (int jj=0; jj<Svv.size();jj++){
            for(int kk=jj; kk<Svv.size();kk++){
                Svv[jj][kk] += vect[jj][ii] * vect[kk][ii];
            }
           vect_mean[jj][start_index] += vect[jj][ii]/w;
        }
    }
    for (int jj=0; jj<Svv.size();jj++){ 

        vect_var[jj][start_index] = Svv[jj][jj]/w - vect_mean[jj][start_index]*vect_mean[jj][start_index];

        for(int kk=jj; kk<Svv.size();kk++){
            cov_vv[jj][kk] = Svv[jj][kk]/w - vect_mean[jj][start_index] *vect_mean[kk][start_index];
        }
    }

    for (int jj=0; jj<Svv.size();jj++){ 
        arr_out[jj][jj][start_index] = 1;
        for(int kk=jj+1; kk<Svv.size();kk++){
            denom = sqrt(vect_var[jj][start_index]  * vect_var[kk][start_index] );
            
            arr_out[jj][kk][start_index] = (denom > 0) ? cov_vv[jj][kk]/denom : 0.0;
            arr_out[kk][jj][start_index] = arr_out[jj][kk][start_index];
            if (abs(arr_out[kk][jj][start_index])>1 ) {
                printf("errore larger than 1.\n");
                cout<< cov_vv[jj][kk]<< " "<< vect_var[jj][start_index] << " "<< vect_var[kk][start_index]<< " "<< jj<< " "<< kk<<endl;
            }
        }
    }

    // DP part for the rolling window
    for (int ii = start_index+1; ii <= end_index-w; ii++){

        {
            int jj=0;
            arr_out[jj][jj][ii] = 1.0;
            Svv[jj][jj]      += vect[jj][ii+w-1] * vect[jj][ii+w-1] - vect[jj][ii-1] * vect[jj][ii-1];
            vect_mean[jj][ii] = (w*vect_mean[jj][ii-1] - vect[jj][ii-1] + vect[jj][ii+w-1])/w;
            vect_var[jj][ii]  = Svv[jj][jj]/w - vect_mean[jj][ii] * vect_mean[jj][ii];

            for(int kk=jj+1; kk<Svv.size();kk++){
                Svv[jj][kk] += vect[jj][ii+w-1] * vect[kk][ii+w-1] - vect[jj][ii-1] * vect[kk][ii-1];
                Svv[kk][kk] += vect[kk][ii+w-1] * vect[kk][ii+w-1] - vect[kk][ii-1] * vect[kk][ii-1];
                
                vect_mean[kk][ii] = (w*vect_mean[kk][ii-1] - vect[kk][ii-1] + vect[kk][ii+w-1])/w;
                cov_vv[jj][kk]   = Svv[jj][kk]/w - vect_mean[jj][ii] * vect_mean[kk][ii];
                vect_var[kk][ii] = Svv[kk][kk]/w - vect_mean[kk][ii] * vect_mean[kk][ii];

                denom = sqrt(vect_var[jj][ii] * vect_var[kk][ii]);

                arr_out[jj][kk][ii] = (denom > 1e-11) ? cov_vv[jj][kk]/denom : 0.0;
                arr_out[kk][jj][ii] = arr_out[jj][kk][ii];
            }
        }

        for (int jj=1; jj<Svv.size();jj++){
            arr_out[jj][jj][ii] = 1.0;
            for(int kk=jj+1; kk<Svv.size();kk++){

                Svv[jj][kk] += vect[jj][ii+w-1] * vect[kk][ii+w-1] - vect[jj][ii-1] * vect[kk][ii-1];
                cov_vv[jj][kk] = Svv[jj][kk]/w - vect_mean[jj][ii] * vect_mean[kk][ii];

                denom = sqrt(vect_var[jj][ii] * vect_var[kk][ii]);

                arr_out[jj][kk][ii] = (denom > 1e-11) ? cov_vv[jj][kk]/denom : 0.0;
                arr_out[kk][jj][ii] = arr_out[jj][kk][ii];

                if (abs(arr_out[kk][jj][ii])>1 ) {
                    printf("error in the algorithm, correlation larger than 1.\n");
                    cout<< cov_vv[jj][kk]<< " "<< vect_var[jj][ii] << " "<< vect_var[kk][ii]<< " "<< jj<< " "<< kk<<endl;
                }
            }
        }
    }
    return;
}


void rolling_var_exec(const vector<vector<double>> &arr_in, vector<vector<double>> &arr_mean, 
                        vector<vector<double>> &arr_var,
                         size_t &w, int start_index, int end_index, int vect_start, int vect_end){

    double sum, mu;

    if (end_index == -1) end_index = (int)arr_in[0].size();
    if (vect_end  == -1) vect_end  = (int)arr_in.size();

    auto lamb = [&mu] (double aa) {return (aa-mu)*(aa-mu);};
    // 1. compute the first var
    for (int jj=vect_start; jj<vect_end;jj++){

        sum    = std::accumulate(arr_in[jj].begin()+start_index, arr_in[jj].begin()+start_index+w, 0.0);
        mu     = sum/w;

        double sum_dev_sq = std::transform_reduce(arr_in[jj].begin() + start_index,arr_in[jj].begin() + start_index + w, 
                                                0.0, std::plus<>(), lamb);

        arr_var[jj][start_index]    = sum_dev_sq/w;
        arr_mean[jj][start_index]   = mu;

    }    

    // 3. Computes the other averages using DP
    //    Iterate from the starting index + 1 until the last available point   
    for (int jj=vect_start; jj<vect_end;jj++){
        double sum_dev_sq = arr_var[jj][start_index] * w;
        for (int ii=start_index+1; ii<=end_index-w;ii++){

            arr_mean[jj][ii] = arr_mean[jj][ii-1] + (arr_in[jj][ii+w-1] - arr_in[jj][ii-1])/w;

            sum_dev_sq = sum_dev_sq - (arr_in[jj][ii-1] - arr_mean[jj][ii]) * (arr_in[jj][ii-1] - arr_mean[jj][ii]) +
                                         + (arr_in[jj][ii+w-1] - arr_mean[jj][ii-1]) * (arr_in[jj][ii+w-1] - arr_mean[jj][ii-1]) ;          

            arr_var[jj][ii]  = sum_dev_sq/w;

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
    vector<double>* arr_out_1D;
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

    if (state->method == "correlation_1Dout")  rolling_mean_corr_exec_mv_1Dout(*state->vect, *state->vect_mean, *state->vect_var, *state->arr_out_1D, state->w, state->start_index, state->end_index);

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
                           vector<double> &arr_out,
                           size_t &w, int num_threads) {

    if (vect.size() < 2)  throw std::runtime_error("rolling_corr_parallel: need at least 2 vectors");
    if (num_threads <= 0) throw std::runtime_error("rolling_corr_parallel: num_threads must be > 0");

    string method = "correlation_1Dout";


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
        args[jj].arr_out_1D = &arr_out;
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