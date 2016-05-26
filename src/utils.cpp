#pragma once

#include <sys/stat.h>
#include <unistd.h>
#include <iostream>
#include <fstream>
#include <iomanip>

#include <thrust/host_vector.h>

using namespace std;

// Printing table for times
// Start with TABLE(precision), then each row is TABLER() followed by TABLEC()...

#define TABLE(p) cout << fixed << setprecision(p)
#define TABLER(width, arg) cout << left << setw(width) << (arg) << ":"
#define TABLEC(width, arg) right << setw(width) << (arg)

unsigned short int double_to_usi(const double f) {
  if (f<0. || f>1.) cerr << "Out of range " << f;
  int i = f * 65535;
  i = i < 0 ? 0 : i;
  i = i > 65535 ? 65535 : i;
  return static_cast<unsigned short int>(i);
}

double usi_to_double (const unsigned short int i) {
  if (i>65535) cerr << "Out of range " << i;
  double f = static_cast<double>(i) / 65535.0;
  return f;
}

// Read ascii doubles as USIs
template <typename Coord_T, size_t Dim>
int read_ascii_points(const char* filename, thrust::host_vector<Coord_T>* pts) {
  *pts = thrust::host_vector<Coord_T>();
  
  ifstream file(filename);
  double c;
  while (file >> c) {
    pts->push_back(double_to_usi(c));
  }
  return pts->size() / Dim;
}

// Read binary USIs
template <typename Coord_T, size_t Dim>
int read_bin_points(const char* filename, thrust::host_vector<Coord_T>* pts) {
  const int csize = sizeof(Coord_T);
  struct stat buf;
  int ret = stat(filename, &buf);
  if (ret < 0) throw "Can't stat file";
  
  const int pts_size(buf.st_size);
  const int npts(pts_size / (Dim * csize));
  
  ifstream stream(filename, ios::binary);
  if (!stream) throw "Can't open file";
  
  *pts = thrust::host_vector<Coord_T>(npts*Dim);
  for (int i=0; i<npts*Dim; ++i) {
    stream.read(reinterpret_cast<char*>(&((*pts)[i])), csize);
  }

  return npts;
}
