#pragma once

#include <sys/stat.h>
#include <sys/times.h>
#include <unistd.h>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <time.h>

#include <thrust/host_vector.h>

using namespace std;

// Printing table for times
// Start with TABLE(precision), then each row is TABLER() followed by TABLEC()...

#define TABLE(p) cout << fixed << setprecision(p)
#define TABLER(width, arg) cout << left << setw(width) << (arg) << ":"
#define TABLEC(width, arg) right << setw(width) << (arg)

struct Timer {
  double total_time;
  timespec* times[2];
  int i;

  Timer() {
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
    #ifdef DEBUG
    cout << "Total time through " << msg << " = " << total_time << ", delta="
         << delta << endl;
    #endif
    return delta;
  }
};

template <typename Coord_T>
int read_points(const char* filename, const int csize, const int dim, thrust::host_vector<Coord_T>* pts) {
  struct stat buf;
  int ret = stat(filename, &buf);
  if (ret < 0) throw "Can't stat file";
  
  const int pts_size(buf.st_size);
  const int npts(pts_size / (dim * csize));
  
  ifstream stream(filename, ios::binary);
  if (!stream) throw "Can't open file";
  
  *pts = thrust::host_vector<Coord_T>(npts*3);
  for (int i=0; i<npts*3; ++i) {
    stream.read(reinterpret_cast<char*>(&((*pts)[i])), csize);
  }

  return npts;
}

