// #include "thread_pool.hpp"
// #include "kernels.hpp"
#include <vector>
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
  size_t w         = (argc > 3) ? stoull(argv[3]) : 256;
  size_t nt        = (argc > 4) ? stoull(argv[4]) : 8;
  int    Ntest_speed = (argc > 5) ? stoull(argv[5]) : 1;

  vector<double> x(n), out(n), out2(n);
  fill(out.begin(),out.end(),0.0);
  fill(out2.begin(),out2.end(),0.0);
  
  boost::mt19937 rng(123);
  boost::normal_distribution<> initial_distribution(0,1);
  boost::variate_generator<boost::mt19937&, boost::normal_distribution<>> initial_velocities_generator(rng,initial_distribution);
  
  for (size_t i = 0; i < n; i++){
    x[i] = initial_velocities_generator();
  }

  for (int j = 0; j < Ntest_speed; j++){
    auto start = chrono::high_resolution_clock::now();
    kernels::rolling_mean_serial(x, out, w);
    auto end  = chrono::high_resolution_clock::now();
    auto diff = chrono::duration<double>(end-start).count();

    printf("serial: %f [s] \n",diff);

    {
    start = chrono::high_resolution_clock::now();
    kernels::rolling_mean_parallel(x, out2, w, num_threads);
    end  = chrono::high_resolution_clock::now();
    diff = chrono::duration<double>(end-start).count();
    printf("parallel: %f [s] \n",diff);
    }

  };
  // quick sanity check
  double max_abs = 0.0;
  for (size_t i = 0; i < n; ++i) {
    if (isnan(out[i]) && isnan(out2[i])) continue;
    max_abs = max(max_abs, abs(out[i] - out2[i]));
    // if (out[i] == 0) cout << "out i zero: "<<i<<endl; 
    // if (out2[i] == 0) cout << "out2 i zero: "<<i<<endl; 
  }
  cout << "max_abs_diff: " << max_abs << "\n";

  return 0;
}
