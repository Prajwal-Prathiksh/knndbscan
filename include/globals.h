#ifndef _globals
#define _globals

#include <cstdlib>
#include <iostream>
#include <fstream>
#include <omp.h>
#include <mpi.h>
#include <assert.h>
#include <vector>
#include <string>
#include <cmath>
#include <math.h>
#include <bits/stdc++.h>
#include <time.h>
#include <algorithm>
#include <map>
#include <iterator>



using namespace std;
extern int maxk;
extern int minPts;
extern float eps;
extern float sentinel;
extern int gsize;
extern int num_threads;
extern vector<int> ilabels;
extern vector<int> jlabels;
extern vector<int> II;
extern vector<int> JJ;
typedef int point_int; // number of points

typedef long long int edge_int; // number of edges

struct Edge{
    point_int j; 
    float w;
};

struct fullEdge{
    point_int i;
    point_int j;
    float w;
};

struct Cycle{
    point_int i;
    point_int j;
};
#pragma omp declare reduction(vec_plus : vector<int> : \
                              transform(omp_out.begin(), omp_out.end(), omp_in.begin(), omp_out.begin(), plus<int>())) \
                    initializer(omp_priv = decltype(omp_orig)(omp_orig.size()))
#pragma omp declare reduction (merge_pointint : vector<point_int> : omp_out.insert(omp_out.end(), omp_in.begin(), omp_in.end()))
#pragma omp declare reduction (merge_edgeint : vector<edge_int> : omp_out.insert(omp_out.end(), omp_in.begin(), omp_in.end()))
#pragma omp declare reduction (merge_fullEdge : vector<fullEdge> : omp_out.insert(omp_out.end(), omp_in.begin(), omp_in.end()))
#pragma omp declare reduction (merge_Cycle : vector<Cycle> : omp_out.insert(omp_out.end(), omp_in.begin(), omp_in.end()))

#endif





