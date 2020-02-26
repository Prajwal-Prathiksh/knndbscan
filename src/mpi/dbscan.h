#ifndef _dbscan
#define _dbscan

#include <vector>
#include <string>
using namespace std;

vector<float> redistribute_points(const int N, const int d, const int C, const int istart, const int npoints, vector<float> point_set);

vector<int> kNN_DBSCAN(int N, float eps, int minPts, vector<int> gids, vector<float> distances);

vector<int> kNNgraph_DBSCAN(const int N, const float eps, const int minPts, const int graph_k, const int nfiles, const string name);

void output_graph_multi(const int n, const int k, const vector<int> gids, const vector<float> distances, const string name);

void output_graph(const int N, const int k, const int nfiles, const vector<int> gids, const vector<float> distances, const string name);

void output_labels(const int N, const vector<int> label, const string name);

vector<int> read_labels(const int N, const string name);

#endif






