#include <iostream>
#include <iomanip>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include "nearptd.cu"
#include "utils.cpp"
#include "timer.cpp"

using namespace std;

int main(const int argc, const char* argv[]) {
  typedef unsigned short int Coord_T;
  const int csize = sizeof(Coord_T);
  const int dim = 3;
  const int psize = dim * csize;
  
  if (argc!=5) {
    cerr << "Error, improper number of arguments (5): " << argc;
    cerr << "Usage: ./nearpt ng_factor fixpts qpts pairs" << endl;
    exit(EXIT_FAILURE);
  }
  nearptd::ng_factor = atof(argv[1]);
  if (nearptd::ng_factor < 0.01 || nearptd::ng_factor > 100) {
    cerr << "Error, illegal ng_factor (" << nearptd::ng_factor << "), set to 1." << endl;
    nearptd::ng_factor = 1.0;
  }
  
  Timer timer = Timer();
  #ifdef PROFILE
  timer.print = true;
  #endif

  thrust::host_vector<Coord_T> fixpts;
  thrust::host_vector<Coord_T> qpts;
  // Read fixed and query points into memory
  const int nfixpts(read_bin_points<Coord_T, dim>(argv[2], &fixpts));
  const int nqpts(read_bin_points<Coord_T, dim>(argv[3], &qpts));
  
  ofstream pstream(argv[4], ios::binary);
  if (!pstream) { 
    throw "ERROR: can't open output file pairs";
  }

  const double time_init = timer("Initialization and reading fixed points");

  // Copy points to GPU
  nearptd::Point_Vector<Coord_T, dim> p(nfixpts, fixpts);
  nearptd::Point_Vector<Coord_T, dim> q(nqpts, qpts);

  const double time_copy = timer("Copying points to GPU");
  
  // Preprocess points into grid
  nearptd::Grid_T<Coord_T, dim> *g = nearptd::Preprocess(nfixpts, &p);

  const double time_fixed = timer("Processing fixed points");

  thrust::host_vector<int> closest(nqpts, -1);
  nearptd::Query(g, &q, &closest);
  
  const double time_query = timer("Querying points");

  // Write points to disk
  for (int i=0; i<nqpts; ++i) {
    pstream.write(reinterpret_cast<char*>(&(qpts[i*3])), psize);
    pstream.write(reinterpret_cast<char*>(&(fixpts[closest[i]*3])), psize);
    #ifdef EXHAUSTIVE
    int close2 = g->exhaustive_query(q[i]);
    if (close2 != closest[i]) {
      cout << "ERROR: (" << i << ") " << closest[i] << " != " << close2 << endl;
    }
    #endif
  }

  const double time_write = timer("Writing points");
  
  const double tf = 1e6 * time_fixed / nfixpts;
  const double tq = 1e6 * time_query / nqpts;

  #ifdef STATS
  cout << fixed << setprecision(5);
  cout << "ng factor: " << nearptd::ng_factor << endl;
  cout << "ng: " << g->ng << endl;
  
  cout << "Mininum points per cell: " << g->Min_Points_Per_Cell << endl;
  cout << "Maximum points per cell: " << g->Max_Points_Per_Cell << endl;
  cout << "Average points per cell: " << g->Avg_Points_Per_Cell << endl;
  cout << "Histogram of number of points per cell:" << endl;
  for (int i=0; i<=g->Max_Points_Per_Cell; ++i) {
    int num = thrust::count(g->Num_Points_Per_Cell.begin()+1, g->Num_Points_Per_Cell.end(), i);
    if (num > 0) {
      cout << i << "\t" << num << "\n";
    }
  }
  cout << endl;
  
  cout << "Total number of queries:      " << nqpts << endl;
  cout << "Number of Fast Case Queries:  " << g->Num_Fast_Queries << endl;
  cout << "Number of Slow Case Queries:  " << g->Num_Slow_Queries << endl;
  cout << "Number of Exhaustive Queries: " << g->Num_Exhaustive_Queries << endl;
  cout << endl;
  #endif

  #ifdef TIMING
  cout << fixed << setprecision(6);
  cout << nfixpts << "\t" << time_init << "\t" << time_copy << "\t"
       << time_fixed << "\t" << time_query << "\t" << time_write << "\t"
       << tf << "\t" << tq << "\t" << timer.total_time << endl;
  #endif
  #ifndef TIMING
  TABLE(4);
  TABLER(20, "nfixpts") << TABLEC(10, nfixpts) << endl;
  TABLER(20, "ng_factor") << TABLEC(10, nearptd::ng_factor) << endl;
  TABLER(20, "grid ng") << TABLEC(10, g->ng) << endl;
  TABLER(20, "init time") << TABLEC(10, time_init) << " (s)" << endl;
  TABLER(20, "copy time") << TABLEC(10, time_copy) << " (s)" << endl;
  TABLER(20, "fixed time") << TABLEC(10, time_fixed) << " (s)" << endl;
  TABLER(20, "query time") << TABLEC(10, time_query) << " (s)" << endl;
  TABLER(20, "write time") << TABLEC(10, time_write) << " (s)" << endl;
  TABLER(20, "time per fixed point") << TABLEC(10, tf) << " (us)" << endl;
  TABLER(20, "time per query point") << TABLEC(10, tq) << " (us)" << endl;
  TABLER(20, "total time") << TABLEC(10, timer.total_time) << " (s)" << endl;
  #endif
  
  return 0;
}
