#include "../include/globals.h"
using namespace std;

void update_cutedges(point_int n0, point_int ISTART, const vector<point_int> R, const map<point_int, int> label, const point_int *JA, const float *A, vector<edge_int> &crossE)
{

    vector<point_int> sbuf, rbuf, ordering;
    int* scounts = new int[gsize];
    int* sdispls = new int[gsize];
    int* rcounts = new int[gsize];
    int* rdispls = new int[gsize];
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int nE = crossE.size();
    int i, j, k, pointer;
    II.resize(nE);
    JJ.resize(nE);
    //prepare send
    int buckets = num_threads * gsize;
    vector<int> counts(buckets, 0);

    #pragma omp parallel 
    {
       int tid = omp_get_thread_num();
    #pragma omp for schedule(static)
    for (int l=0; l<nE; l++){
        edge_int e = crossE[l];
        int i = floor((double)e/(double)maxk);
        int j = JA[e];
        int k = floor((double)j/(double)n0);
        II[l] = label.at(R[i]);
        int kk = k*num_threads+tid;
        counts[kk] += 1;
    }
    }

    vector<int> displs(buckets, 0);
    int counter = 0;
    int position;
    for (int l = 0; l<gsize; l++){
         sdispls[l] = (l==0) ? 0:sdispls[l-1] + scounts[l-1];
         scounts[l] = 0;
         for(int m = 0; m<num_threads; m++){
            position = l*num_threads + m;
            displs[position] = counter;
            scounts[l] += counts[position];
            counter += counts[position];
         }
    }

    sbuf.resize(nE);
    ordering.resize(nE);
    fill(counts.begin(), counts.end(), 0);
    #pragma omp parallel 
    {
       int tid = omp_get_thread_num();
    #pragma omp for schedule(static)
    for(int l =0; l<nE; l++){
        int j = JA[crossE[l]];
        int k = floor((double)j/(double)n0);
        int kk = k*num_threads+tid;
        int pointer = displs[kk]+counts[kk];
        sbuf[pointer] = j;
        ordering[l] = pointer;
        counts[kk] += 1;
    }
    }

    //1st all to all
    MPI_Alltoall(scounts, 1, MPI_INT, rcounts, 1, MPI_INT, MPI_COMM_WORLD);
    k = 0;
    for(int l = 0; l < gsize; l++) {
        rdispls[l] = (l==0) ? 0:rdispls[l-1] + rcounts[l-1];
        k += rcounts[l];
    }
    rbuf.resize(k);
    MPI_Alltoallv(&sbuf[0], scounts, sdispls, MPI_INT, &rbuf[0], rcounts, rdispls, MPI_INT, MPI_COMM_WORLD);

    //update and 2nd all to all to send back new pointer
    #pragma omp parallel for
    for(int l = 0; l< k; l++)
    {
        int j = rbuf[l] - ISTART;
        rbuf[l] = (R[j] < 0) ? -1:label.at(R[j]);
    }
    MPI_Alltoallv(&rbuf[0], rcounts, rdispls, MPI_INT, &sbuf[0], scounts, sdispls, MPI_INT, MPI_COMM_WORLD);
    #pragma omp parallel for
    for(int l = 0; l < nE; l++)
    {
        int pointer = ordering[l];
        int j = sbuf[pointer];
        if(j >= 0){
            JJ[l] = j;
        }else{
            crossE[l] = (edge_int) -1;
        }
    }
}







