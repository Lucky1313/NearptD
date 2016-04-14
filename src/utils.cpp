#include <sys/stat.h>
#include <sys/times.h>
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


float clocks_per_sec = -1.0;	// will be set later.
double total_time;

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
  //cout << "\n.....Total CPU Time thru " << msg << " = " << total_time << ", increment="
  //     << incrtime << '\n' << endl;
  return incrtime;
}

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
