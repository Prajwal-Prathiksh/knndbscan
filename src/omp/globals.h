#ifndef _globals
#define _globals
using namespace std;

typedef int point_int; // number of points

typedef long long int edge_int; // number of edges

struct Edge{
    point_int j; 
    float w;
};

struct Cycle{
    point_int i;
    point_int j;
};
#endif





