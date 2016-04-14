#include <iostream>
#include <iomanip>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include "nearpt3.cu"
#include "utils.cpp"

int main(const int argc, const char* argv[]) {
  typedef unsigned short int Coord_T;
  const int csize = sizeof(Coord_T);
  const int psize = 3 * csize;
  
  if (argc!=5) {
    cerr << "Error, improper number of arguments (5): " << argc;
    cerr << "Usage: ./nearpt ng_factor fixpts qpts pairs" << endl;
    exit(EXIT_FAILURE);
  }
  nearpt3::ng_factor = atof(argv[1]);
  if (nearpt3::ng_factor < 0.01 || nearpt3::ng_factor > 100) {
    cerr << "Error, illegal ng_factor (" << nearpt3::ng_factor << "), set to 1." << endl;
    nearpt3::ng_factor = 1.0;
  }
  
  Print_Time("Init");

  thrust::host_vector<Coord_T> fixpts;
  thrust::host_vector<Coord_T> qpts;
  
  // Read fixed and query points into memory
  const int nfixpts(read_points<Coord_T>(argv[2], csize, 3, &fixpts));
  const int nqpts(read_points<Coord_T>(argv[3], csize, 3, &qpts));

  const double time_init = Print_Time("Initialization and reading fixed points");
  
  // Structure of arrays rather than Array of structures, thrust good practice
  // Contains 3 device vectors, one for x, y, z
  nearpt3::Point_Vector<Coord_T> *p = new nearpt3::Point_Vector<Coord_T>(nfixpts, fixpts);
  nearpt3::Point_Vector<Coord_T> *q = new nearpt3::Point_Vector<Coord_T>(nqpts, qpts);

  const double time_copy = Print_Time("Copying points to GPU");
      
  nearpt3::Grid_T<Coord_T> *g = nearpt3::Preprocess(nfixpts, p);

  const double time_fixed = Print_Time("Processing fixed points");
  
  ofstream pstream(argv[4], ios::binary);
  if (!pstream) { 
    throw "ERROR: can't open output file pairs";
  }

  thrust::host_vector<int> closest;
  nearpt3::Query<Coord_T>(g, q, &closest);
  for (int i=0; i<nqpts; ++i) {
    pstream.write(reinterpret_cast<char*>(&(qpts[i*3])), psize);
    pstream.write(reinterpret_cast<char*>(&(fixpts[closest[i]*3])), psize);
    #ifdef EXHAUSTIVE
    int close2 = g->exhaustive_query((*q)[i]);
    if (close2 != closest[i]) {
      cout << "ERROR: " << closest[i] << " != " << close2 << endl;
    }
    #endif
  }

  const double time_query = Print_Time("Querying points");
  const double tf = 1e6 * (time_fixed + time_copy) / nfixpts;
  const double tq = 1e6 * time_query / nqpts;

  #ifdef STATS
  cout << fixed << setprecision(5);
  cout << "ng factor: " << nearpt3::ng_factor << endl;
  cout << "ng: " << g->ng << endl;
  
  cout << "Mininum points per cell: " << g->Min_Points_Per_Cell << endl;
  cout << "Maximum points per cell: " << g->Max_Points_Per_Cell << endl;
  cout << "Average number of points per cell: " << g->Avg_Points_Per_Cell << endl;
  cout << "Histogram of number of points per cell:" << endl;
  for (int i=0; i<=g->Max_Points_Per_Cell; ++i) {
    int num = thrust::count(g->Num_Points_Per_Cell.begin()+1, g->Num_Points_Per_Cell.end(), i);
    if (num > 0) {
      cout << i << "\t" << num << "\n";
    }
  }
  cout << endl;
  
  cout << "Total number of queries: " << nqpts << endl;
  cout << "Number of Fast Case Queries: " << g->Num_Fast_Queries << endl;
  cout << "Number of Slow Case Queries: " << g->Num_Slow_Queries << endl;
  cout << "Number of Exhaustive Queries: " << g->Num_Exhaustive_Queries << endl;
  cout << endl;

  /*
  cout << "Total number of cells searched: " << g->Total_Cells_Searched << endl;
  cout << "Average number of cells searched per query: " <<
    static_cast<float>(g->Total_Cells_Searched) / static_cast<float>(nqpts) << endl;
  cout << "Histogram of number of queries that searched a given number of cells (up to " << g->Max_Cells_Searched << "):" << endl;
  for (int i=0; i<g->Max_Cells_Searched; ++i) {
    if (g->Num_Cells_Searched[i] > 0) {
      cout << i << "\t" << g->Num_Cells_Searched[i] << "\n";
    }
  }
  if (g->Num_Cells_Searched[g->Max_Cells_Searched] > 0) {
    cout << g->Max_Cells_Searched << "+\t" << g->Num_Cells_Searched[g->Max_Cells_Searched] << endl;
  } 
  cout << endl;
  
  cout << "Total number of points checked: " << g->Total_Points_Checked << endl;
  cout << "Average number of points checked per query: " <<
    static_cast<float>(g->Total_Points_Checked) / static_cast<float>(nqpts) << endl;
  cout << "Histogram of number of queries that searched a given number of points (up to " << g->Max_Points_Checked << "):" << endl;
  for (int i=0; i<g->Max_Points_Checked; ++i) {
    if (g->Num_Points_Checked[i] > 0) {
      cout << i << "\t" << g->Num_Points_Checked[i] << "\n";
    }
  }
  if (g->Num_Points_Checked[g->Max_Points_Checked] > 0) {
    cout << g->Max_Points_Checked << "+\t" << g->Num_Points_Checked[g->Max_Points_Checked] << endl;
  }
  cout << endl;
  */
  #endif

  #ifdef TIMING
  cout << fixed << setprecision(8);
  cout << nfixpts << "\t" << time_init << "\t" << (time_copy + time_fixed) <<
    "\t" << time_query << "\t" << tf << "\t" << tq << "\t" << total_time << endl;
  #endif
  #ifndef TIMING
  TABLE(4);
  TABLER(20, "nfixpts") << TABLEC(10, nfixpts) << endl;
  TABLER(20, "ng_factor") << TABLEC(10, nearpt3::ng_factor) << endl;
  TABLER(20, "grid ng") << TABLEC(10, g->ng) << endl;
  TABLER(20, "init time") << TABLEC(10, time_init) << " (s)" << endl;
  TABLER(20, "copy time") << TABLEC(10, time_copy) << " (s)" << endl;
  TABLER(20, "fixed time") << TABLEC(10, time_fixed) << " (s)" << endl;
  TABLER(20, "query time") << TABLEC(10, time_query) << " (s)" << endl;
  TABLER(20, "time per fixed point") << TABLEC(10, tf) << " (us)" << endl;
  TABLER(20, "time per query point") << TABLEC(10, tq) << " (us)" << endl;
  TABLER(20, "total time") << TABLEC(10, total_time) << " (s)" << endl;
  #endif
  
  return 0;
}
