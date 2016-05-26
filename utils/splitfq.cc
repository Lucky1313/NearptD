// Time-stamp: </wrf/c/nearpt3/splitfq.cc, Sun,  1 May 2005, 21:22:47 EDT, http://wrfranklin.org/>

// Read points, an ascii file of 3D floating points, scaled to be in [0,1].  Select 10K equally
// spaced points to write to file qpts.  Write the rest to fixpts.  Since the total number of
// points is unkown, read all the points twice: once to count and once to select.

// Write the output files in binary, 3 unsigned short ints per point, scaled to be in [0,65535].

// W. Randolph Franklin
// nearpt3 AT wrfranklin.org (Plaintext preferred; attachments deprecated)
// http://wrfranklin.org

#include <iostream>
#include <fstream>
#include <boost/multi_array.hpp> 
#include <math.h>
#include <time.h>
#include <sys/mman.h>
#include <sys/time.h>
#include <sys/times.h>
#include <sys/types.h>

using namespace std;
using boost::array;

// Print an expression's name then its value, possible followed by a comma or endl.  
// Ex: cout << PRINTC(x) << PRINTN(y);

#define PRINT(arg)  #arg "=" << (arg)
#define PRINTC(arg)  #arg "=" << (arg) << ", "
#define PRINTN(arg)  #arg "=" << (arg) << endl


typedef array<unsigned short int,3> Pt_T;
Pt_T p;

float fp[3];

unsigned short int conv(const float f) {
  if (f<0. || f > 1.) cerr << "Out of range " << PRINTN(f);
  int i = f * 65535;
  if (i<0) i=0;
  if (i>65535) i=65535;
  return static_cast<unsigned short int>(i);
}

int main(const int argc, const char* argv[]) {

  if (argc < 5) {
    cerr << "Incorrect number of arguments" << endl;
    exit(1);
  }
  
  int np(0);   // Number of input points
  int iq(0);   // Number of query points found so far.
  
  const int nq(atoi(argv[1]));  // Desired number of output query points.

  const int sp = sizeof(p);
  cerr << PRINTN(sp);

  {
    FILE *sin;
    sin = fopen64(argv[2],"r");
    cerr << "Reading points the 1st time: ";
    if (!sin) {perror("fopen failed the first time");exit(1);}
    for (;;) {
      int ret=fscanf(sin,"%f %f %f",&fp[0], &fp[1], &fp[2]);
      if (ret<=0) break;
      if (0==(np++%1000000)) { cerr << np << ' '; }
    }
    cerr << PRINTN(np);
    if (np<nq) { cerr << "np is too small to select " << PRINTC(nq) << " points.  Giving up." << endl;
      exit(1);
    }
  }
  {
    FILE *sin;
    sin = fopen64(argv[2],"r");
    cerr << "Reading points the 2nd time: ";
    if (!sin) {perror("fopen failed the 2nd time");exit(1);}
    ofstream fstream(argv[3],ios::binary);   // ios::binary unnecessary in linux.
    if (!fstream) { 
      cout << "ERROR: can't open file fixpts" << endl;
      exit(1);
    }
    ofstream qstream(argv[4],ios::binary);
    if (!qstream) { 
      cout << "ERROR: can't open file qpts" << endl;
      exit(1);
    }

    // This will write exactly nq query points.

    //  char s[np];
    char *s;
    s = new char[np];

    for (int i=0; i<np; i++) s[i]=0;
    const double r = static_cast<double>(np)/static_cast<double>(nq);
    for (int i=0; i<nq; i++) s[static_cast<int>(i*r)] = 1;   // i*r truncates.

    for (int ip=0; ip<np; ip++) {
      int ret=fscanf(sin,"%f %f %f",&fp[0], &fp[1], &fp[2]);
      if (ret>0) {
	for (int i=0; i<3; i++) p[i] = conv(fp[i]);
	if (s[ip]) qstream.write(reinterpret_cast<char*>(&p),sp);
	else fstream.write(reinterpret_cast<char*>(&p),sp);
      } else {
	cerr << "Unexpected failure when rereading pts, at " << PRINTN(ip);
	exit(1);
      }
    }
  }
}



