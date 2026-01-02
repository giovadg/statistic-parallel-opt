// #include "thread_pool.hpp"
// #include "kernels.hpp"
#include <vector>
#include <tuple>
#include <random>
#include <chrono>
#include <iostream>
#include <cstdlib>
#include <boost/random.hpp>
#include <unistd.h>
#include "kernels.hpp"


using namespace std; 

int main(int argc, char** argv) {
  size_t n         = (argc > 1) ? stoull(argv[1]) : 5000;
  int num_threads  = (argc > 2) ? stoull(argv[2]) : 1;
  size_t w         = (argc > 4-1) ? stoull(argv[4-1]) : n/5;
  int    Ntest_speed = (argc > 5-1) ? stoull(argv[5-1]) : 1;

  vector<double> x1(n), x2(n);
  vector<double> roll_av_x1_ser(n), roll_av_x1_pll(n),  roll_av_x2_ser(n), roll_av_x2_pll(n); 
  vector<double> roll_corr_ser(n), roll_corr_pll(n); 

  string method;
  unordered_map<string,double> timings;
  
  vector<vector<double>> x_tot;
  vector<vector<double>> roll_av_xtot_pll(2,vector<double>(n));
  
  // Random number generators: 2 Gaussians, with different mean and std. Different seeds.
  boost::mt19937 rng1(123);
  boost::mt19937 rng2(100);
  boost::normal_distribution<> initial_distribution1(0,1);
  boost::normal_distribution<> initial_distribution2(1,2);

  boost::variate_generator<boost::mt19937&, boost::normal_distribution<>> 
  initial_velocities_generator1(rng1,initial_distribution1);

  boost::variate_generator<boost::mt19937&, boost::normal_distribution<>> 
  initial_velocities_generator2(rng2,initial_distribution2);
  
  for (size_t i = 0; i < n; i++){
    x1[i] = initial_velocities_generator1();
    x2[i] = initial_velocities_generator2();
  }
  auto tuple_vect = std::make_tuple(x1, x2); 

  x_tot.emplace_back(x1);
  x_tot.emplace_back(x2);
  roll_av_xtot_pll.emplace_back(roll_av_x1_pll);
  roll_av_xtot_pll.emplace_back(roll_av_x2_pll);

  for (int j = 0; j < Ntest_speed; j++){

    auto start = chrono::high_resolution_clock::now();

    kernels::rolling_mean_exec(x1, roll_av_x1_ser, w);
    kernels::rolling_mean_exec(x2, roll_av_x2_ser, w);

    auto tuple_mean = std::make_tuple(roll_av_x1_ser, roll_av_x2_ser); 

    // computes both mean and correlation
    kernels::rolling_mean_corr_exec(tuple_vect, tuple_mean, roll_corr_ser, w);

    auto end  = chrono::high_resolution_clock::now();
    auto diff = chrono::duration<double>(end-start).count();
    method = "serial vectors input - serial vector treatment:";
    timings[method] = (timings.find(method) != timings.end())? timings[method] + diff : timings[method] = diff;

    // --------------------------
    // parallel approach: division of arrays in subarrays.
    start = chrono::high_resolution_clock::now();
    kernels::rolling_mean_parallel(x1, roll_av_x1_pll, w, num_threads);
    kernels::rolling_mean_parallel(x2, roll_av_x2_pll, w, num_threads);
    tuple_mean = std::make_tuple(roll_av_x1_ser, roll_av_x2_ser);
    kernels::rolling_corr_parallel(tuple_vect, tuple_mean, roll_corr_pll, w, num_threads);
    end  = chrono::high_resolution_clock::now();
    diff = chrono::duration<double>(end-start).count();
    method = "serial vectors input - parallel vector treatment:";
    timings[method] = (timings.find(method) != timings.end())? timings[method] + diff : timings[method] = diff;


    // parallel approach: parallelized on the different arrays
    // if nested: each array divided also in subarray - not applicable for correlation -
    for (bool nested_threads : {false,true}){
    start = chrono::high_resolution_clock::now();
    kernels::rolling_mean_parallel_inputs(x_tot, roll_av_xtot_pll, w, num_threads, nested_threads);
    tuple_mean = std::make_tuple(roll_av_x1_ser, roll_av_x2_ser);
    kernels::rolling_corr_parallel(tuple_vect, tuple_mean, roll_corr_pll, w, num_threads);

    end  = chrono::high_resolution_clock::now();
    diff = chrono::duration<double>(end-start).count();
    if (nested_threads){
      method="parallel vectors input - parallel vector treatment:";// %f [s] \n",diff);
    }else{
      method="parallel vectors input - serial vector treatment:";// %f [s] \n",diff);
    }
    timings[method] = (timings.find(method) != timings.end())? timings[method] + diff : timings[method] = diff;
    }

  };
  for (auto it=timings.begin(); it != timings.end();it++){
    cout << it->first <<" "<< (it->second)/Ntest_speed<<" [s]"<<endl;
    cout <<" "<<endl;
  }

  // quick sanity check
  vector<double> max_abs(5);
  for (size_t i = 0; i < n; ++i) {
    if (isnan(roll_av_x1_ser[i]) && isnan(roll_av_x1_pll[i])) continue;
    max_abs[0] = max(max_abs[0], abs(roll_av_x1_ser[i] - roll_av_x1_pll[i]));
    if (isnan(roll_av_x2_ser[i]) && isnan(roll_av_x2_pll[i])) continue;
    max_abs[1] = max(max_abs[1], abs(roll_av_x2_ser[i] - roll_av_x2_pll[i]));
    max_abs[2] = max(max_abs[2], abs(roll_av_x1_ser[i] - roll_av_xtot_pll[0][i]));
    max_abs[3] = max(max_abs[3], abs(roll_av_x2_ser[i] - roll_av_xtot_pll[1][i]));
    max_abs[4] = max(max_abs[4], abs(roll_corr_ser[i] - roll_corr_pll[i]));

  }
  cout <<"comparison between serial and single layer parallelism: \nmax_abs_diff_1: " << max_abs[0] <<  "\nmax_abs_diff_2: " << max_abs[1] <<"\n";
  cout << "comparison between serial and nested parallelism: \nmax_abs_diff_1: " << max_abs[2] <<  "\nmax_abs_diff_2: " << max_abs[3] <<"\n";
  cout << "Max correlation difference between serial and parallel: " << max_abs[4] <<"\n";
  cout <<"max correlation is: " << *max_element(roll_corr_ser.begin(), roll_corr_ser.end());
  return 0;
}
