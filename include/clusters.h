#ifndef CLUSTERS_H
#define CLUSTERS_H

#include "globals.h"

vector<point_int> knndbscan(const point_int N, const float eps_value, const int minPts_value, const int maxk_value, const point_int *JA, const float *A);

#endif