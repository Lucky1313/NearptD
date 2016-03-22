#include <thrust/host_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/iterator/zip_iterator.h>

#include "nearpt3.cu"

float clocks_per_sec = -1.0;	// will be set later.
double total_time;

// Printing table for times at end
// Start with TABLE(precision), then each row is TABLER() followed by TABLEC()...

#define TABLE(p) cout << fixed << setprecision(p)
#define TABLER(width, arg) cout << left << setw(width) << (arg) << ":"
#define TABLEC(width, arg) right << setw(width) << (arg)

// GET_PROCESS_CPU_TIME Return CPU (user+system) time since start of process.

double Get_Process_CPU_Time ()
{
  struct tms *time_buffer = new tms;
  (void) times (time_buffer);

  if (clocks_per_sec<0.0) clocks_per_sec = sysconf (_SC_CLK_TCK);
  return ((time_buffer->tms_utime + time_buffer->tms_stime) / clocks_per_sec);
}


//  GET_DELTA_TIME Returns time in seconds since last Get_Delta_Time.  Automatically initializes
//  itself on 1st call and returns 0.  Also, return time from process start in its arg.

double Get_Delta_Time (double &new_time)
{
  static double old_time = 0.0;
  double  delta;

  new_time = Get_Process_CPU_Time ();
  delta = new_time - old_time;
  old_time = new_time;
  return delta;
}


// PRINT_TIME Print time since last call, with a message.  Then return the incremental time.

double Print_Time (string msg)
{
  double  incrtime = Get_Delta_Time (total_time);
  //  cout << "\n.....Total CPU Time thru " << msg << " = " << total_time << ", increment="
  //       << incrtime << '\n' << endl;
  return incrtime;
}

int main(const int argc, const char* argv[]) {
  typedef unsigned short int Coord_T;
  const int csize = sizeof(Coord_T);
  const int psize = 3*csize;
  struct stat buf;
  
  if (argc<5) {
    cerr << "Error, improper number of arguments: " << PRINTN(argc);
    cerr << "Usage: ./nearpt ng_factor fixpts qpts pairs" << endl;
    exit(1);
  }
  Print_Time("Init");
  
  nearpt3::ng_factor = atof(argv[1]);
  if (nearpt3::ng_factor < 0.01 || nearpt3::ng_factor > 100) {
    cerr << "Error, illegal " << PRINTC(nearpt3::ng_factor) << " set to 1." << endl;
    nearpt3::ng_factor = 1.0;
  }

  int ret = stat(argv[2], &buf);
  if (ret < 0) throw "Can't stat fixpts";
  
  const int fixpts_size = buf.st_size;
  const int nfixpts = fixpts_size / psize;

  #ifdef DEBUG
  cout << "Number of fixed points: " << nfixpts << endl;
  #endif

  ifstream fixstream(argv[2], ios::binary);
  if (!fixstream) {
    throw "Error, can't open file fixpts";
  }

  const double time_read = Print_Time("Read");

  // Host vector for read points, not split into x,y,z
  thrust::host_vector<Coord_T> pts(nfixpts * 3);
  for (int i=0; i<nfixpts*3; ++i) {
    fixstream.read(reinterpret_cast<char*>(&pts[i]), csize);
  }

  const double time_init = Print_Time("Initialization and reading fixed points");

  // Structure of arrays rather than Array of structures, thrust good practice
  // Contains 3 device vectors, one for x, y, z
  nearpt3::Points_Vector<Coord_T> *p  = new nearpt3::Points_Vector<Coord_T>(nfixpts, pts); 

  const double time_copy = Print_Time("Copying points to GPU");
      
  nearpt3::Grid_T<Coord_T> *g = nearpt3::Preprocess(nfixpts, p);

  const double time_fixed = Print_Time("Processing fixed points");

  ifstream qstream(argv[3], ios::binary);
  if (!qstream) { 
    throw "ERROR: can't open file qpts";
  }
  ofstream pstream(argv[4], ios::binary);
  if (!pstream) { 
    throw "ERROR: can't open output file pairs";
  }

  int nqpts = 0;
  array<Coord_T,3> q, pt;
  while (qstream.read(reinterpret_cast<char*>(&q), 3*sizeof(Coord_T))) {
    nqpts++;
    int closestpt = nearpt3::Query(g, q);
    pstream.write(reinterpret_cast<char*>(&q), psize);
    pt[0] = pts[closestpt*3];
    pt[1] = pts[closestpt*3+1];
    pt[2] = pts[closestpt*3+2];
    pstream.write(reinterpret_cast<char*>(&pt), psize);
  }

  const double time_query = Print_Time("Querying points");
  const double tf = 1e6 * (time_fixed + time_copy) / nfixpts;
  const double tq = 1e6 * time_query / nqpts;

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
  
  return 0;
}