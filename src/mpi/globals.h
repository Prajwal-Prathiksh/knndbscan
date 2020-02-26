#ifndef _globals
#define _globals
using namespace std;

typedef long long int edge_int;

struct Edge{
    int i;
    int j;
    float w;
    int e;
};

struct newEdge{
    int i;
    int j;
    float w;
};

struct EdgeSend{
    int j;
    float w;
};

struct Border{
    int b;
    int label;
};

#endif






