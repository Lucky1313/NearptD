#pragma once

#include <iostream>
#include <time.h>

using namespace std;

struct Timer {
  double total_time;
  timespec* times[2];
  int i;
  bool print;

  Timer(bool print=false) : print(print) {
    total_time = 0.0;
    i=0;
    times[0] = new timespec;
    times[1] = new timespec;
    get_time();
  }
  
  void get_time() {
    clock_gettime(CLOCK_REALTIME, times[i]);
    i = (i ? 0 : 1);
  }
  
  double delta_time() {
    get_time();
    timespec* new_time = times[(i ? 0 : 1)];
    timespec* prev_time = times[i];
    double delta = (new_time->tv_sec - prev_time->tv_sec) +
      (new_time->tv_nsec - prev_time->tv_nsec) / 1.0e9;
    total_time += delta;
    return delta;
  }

  double operator()(string msg) {
    double delta = delta_time();
    if (print) {
    cout << "Total time through " << msg << " = " << total_time << ", delta="
         << delta << endl;
    }
    return delta;
  }
};

