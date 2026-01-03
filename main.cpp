
#include <vector>
#include <tuple>
#include <random>
#include <chrono>
#include <iostream>
#include <cstdlib>
#include <boost/random.hpp>
#include <unistd.h>
#include "kernels.hpp"
#include "interface_vectors.hpp"
#include <fstream>

using namespace std; 


unordered_map<string,string> input_dict(int argc, char** argv) {
    unordered_map<string,string> args;
    for (int i=1; i<argc; ++i) {
        string s(argv[i]);
        // auto pos = s.find('=');
        if (s.find('=') == string::npos) continue; 
        args[s.substr(0,s.find('='))] = s.substr(s.find('=')+1);
    }
    return args;
}

int main(int argc, char** argv) {

  unordered_map<string,string> args = input_dict(argc, argv);

  string path      = ((args.find("path") != args.end())) ? string(args["path"]) : "none";
  size_t n         = ((args.find("n") != args.end())) ? stoull(args["n"]) : 5000;
  int num_threads  = ((args.find("num_threads") != args.end())) ? stoull(args["num_threads"]) : 1;
  int Ntest_speed  = ((args.find("Ntest_speed") != args.end())) ? stoull(args["Ntest_speed"]) : 1;


  vector<vector<double>> x_tot;
  vector<vector<double>> roll_av_ser, roll_av_pll;
  vector<vector<double>> roll_corr_ser, roll_corr_pll; 

  generation::interface_vectors_generation(path, n, x_tot, roll_av_ser, roll_av_pll,
                                           roll_corr_ser, roll_corr_pll);

  size_t dim_vectors = x_tot[0].size(); 
  size_t w         = ((args.find("w") != args.end())) ? stoull(args["w"]) : dim_vectors/10;

  cout <<"variables are: "<<num_threads<<" "<<dim_vectors << " "<<w<<" "<<Ntest_speed<<" "<<path<<endl;

  string method;
  unordered_map<string,double> timings;
  
  for (int j = 0; j < Ntest_speed; j++){

    auto start = chrono::high_resolution_clock::now();

    kernels::rolling_mean_exec(x_tot[0], roll_av_ser[0], w);
    kernels::rolling_mean_exec(x_tot[1], roll_av_ser[1], w);


    // computes both mean and correlation
    kernels::rolling_mean_corr_exec(x_tot, roll_av_ser, roll_corr_ser[0], w);

    auto end  = chrono::high_resolution_clock::now();
    auto diff = chrono::duration<double>(end-start).count();
    method = "serial vectors input - serial vector treatment";
    timings[method] = (timings.find(method) != timings.end())? timings[method] + diff : timings[method] = diff;

    // --------------------------
    // parallel approach: division of arrays in subarrays.
    start = chrono::high_resolution_clock::now();
    kernels::rolling_mean_parallel(x_tot[0], roll_av_pll[0], w, num_threads);
    kernels::rolling_mean_parallel(x_tot[1], roll_av_pll[1], w, num_threads);

    kernels::rolling_corr_parallel(x_tot, roll_av_ser, roll_corr_pll[0], w, num_threads);
    end  = chrono::high_resolution_clock::now();
    diff = chrono::duration<double>(end-start).count();
    method = "serial vectors input - parallel vector treatment";
    timings[method] = (timings.find(method) != timings.end())? timings[method] + diff : timings[method] = diff;


    // parallel approach: parallelized on the different arrays
    // if nested: each array divided also in subarray - not applicable for correlation -
    for (bool nested_threads : {false,true}){
    start = chrono::high_resolution_clock::now();
    kernels::rolling_mean_parallel_inputs(x_tot, roll_av_pll, w, num_threads, nested_threads);

    kernels::rolling_corr_parallel(x_tot, roll_av_ser, roll_corr_pll[0], w, num_threads);

    end  = chrono::high_resolution_clock::now();
    diff = chrono::duration<double>(end-start).count();
    if (nested_threads){
      method="parallel vectors input - parallel vector treatment";
    }else{
      method="parallel vectors input - serial vector treatment";
    }
    timings[method] = (timings.find(method) != timings.end())? timings[method] + diff : timings[method] = diff;
    }

  };
  std::ofstream f("timing.csv");
  f << "method"<<";"<<"time"<<"\n";

  for (auto it=timings.begin(); it != timings.end();it++){
    cout << it->first <<": "<< (it->second)/Ntest_speed<<" [s]"<<endl;
    cout <<" "<<endl;
    f << it->first<<";"<<it->second<<"\n";
  }

  // quick sanity check
  vector<double> max_abs(5);
  for (size_t i = 0; i < dim_vectors; ++i) {
    if (isnan(roll_av_ser[0][i]) && isnan(roll_av_pll[0][i])) continue;
    max_abs[0] = max(max_abs[0], abs(roll_av_ser[0][i] - roll_av_pll[0][i]));
    if (isnan(roll_av_ser[1][i]) && isnan(roll_av_pll[1][i])) continue;
    max_abs[1] = max(max_abs[1], abs(roll_av_ser[1][i] - roll_av_pll[1][i]));
    max_abs[2] = max(max_abs[2], abs(roll_av_ser[0][i] - roll_av_pll[0][i]));
    max_abs[3] = max(max_abs[3], abs(roll_av_ser[1][i] - roll_av_pll[1][i]));
    max_abs[4] = max(max_abs[4], abs(roll_corr_ser[0][i] - roll_corr_pll[0][i]));

  }

  if (*max_element(max_abs.begin(),max_abs.end()) > 1e-10){
    cout <<"comparison between serial and single layer parallelism: \nmax_abs_diff_1: "<< max_abs[0] <<  "\nmax_abs_diff_2: " << max_abs[1] <<"\n";
    cout << "comparison between serial and nested parallelism: \nmax_abs_diff_1: " << max_abs[2] <<  "\nmax_abs_diff_2: " << max_abs[3] <<"\n";
    cout << "Max correlation difference between serial and parallel: " << max_abs[4] <<"\n";
    cout <<"max correlation is: " << *max_element(roll_corr_ser[0].begin(), roll_corr_ser[0].end());
  }
  return 0;
}
