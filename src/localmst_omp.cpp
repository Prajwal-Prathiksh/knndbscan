#include "../include/globals.h"
#include "mst_omp.cpp"
using namespace std;

int localmst_omp(const point_int N, const point_int ISTART, const point_int *JA, const float *A, vector<point_int> &R, vector<point_int> &C, const vector<point_int> core, vector<edge_int> &crossE)
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    //note that for multi process, N represents total # points of the local process
    point_int N_core = core.size();
    point_int n_tree = N_core;
    point_int dn_tree = 1;

    vector<point_int> checked(N), cycleFlags(N);
    vector<Edge> minE(N);
    vector<Cycle> cycles;

    vector<int> isteps;
    vector<int> n_trees;

    point_int count;
    vector<point_int> counts(num_threads);
    int istep = 0;
    point_int n_tree_old;
    while(dn_tree > 0){
        //phase-I: initialization
        #pragma omp parallel for
        for(point_int c = 0; c < n_tree; c++)
        {
           point_int i = C[c];
           minE[i] = {-1, sentinel};
        }
        //phase-II: find min_edges
        #pragma code_align 32
        #pragma omp parallel for reduction(merge_edgeint: crossE)
        for(point_int c = 0; c < N_core; c++)
        {
            point_int i = core[c]; // for a core point
            edge_int m = i * maxk + checked[i];
            while(checked[i] < minPts-1){
                point_int q = JA[m] - ISTART;
                if((q < 0) or (q >= N )){ // this a cross edge, need to check the next one
                    crossE.push_back(m);
                    checked[i] += 1;
                    m += 1;
                }else if((R[q] > -1) and (R[i] != R[q])){ //this edge is internal edge connecting two core points
                    pwrite(minE[R[i]], R[q], A[m]);
                    break;
                }else{ //internal but connecting to a non-core point, need to check the next one
                    checked[i] += 1;
                    m += 1;
                }
            }
        }
        //phase-III: break symmetry
        #pragma omp parallel for
        for(point_int c = 0; c < n_tree; c++)
        {
            point_int i = C[c];
            point_int j = minE[i].j;
            if(j == -1){
                minE[i].j = i;
            }else{
                point_int k = minE[j].j;
                if((i == k) and (i < j)){
                    minE[i].j = i;
                }
            }
        }
        //phase-IV: pointer jumping
        #pragma omp parallel for
        for(point_int c = 0; c < n_tree; c++)
        {
            point_int i = C[c];
            cycleFlags[i] = minE[i].j;
        }

        count = 1;
        while(count > 0) {
            count = 0;
            #pragma omp parallel for
            for(point_int c = 0; c < n_tree; c++)
            {
                point_int i = C[c];
                point_int j = minE[i].j;
                point_int k = minE[j].j;
                if(j!=k){
                    minE[i].j = k;
                }else if(cycleFlags[i] >= 0){
                    cycleFlags[i] = -1;
                    counts[omp_get_thread_num()] += 1;
                }
            }
            for(int t = 0; t<num_threads; t++) count += counts[t];
            fill(counts.begin(), counts.end(), 0);
        }
        //phase-V: break cycles
        cycles.resize(0);
        #pragma omp parallel for reduction(merge_Cycle: cycles)
        for(point_int c = 0; c<n_tree; c++)
        {
            point_int i = C[c];
            if(cycleFlags[i]!=-1){
                cycles.push_back({i, cycleFlags[i]});
            }
        }
        if(cycles.size() > 0){
            break_cycles(cycles, minE);
        }
        //phase-VI: update roots R, number of trees and clusters
        n_tree_old = n_tree;
        n_tree = 0;
        C.resize(0);
        #pragma omp parallel for reduction(merge_pointint: C)
        for(point_int c = 0; c < N_core; c++)
        {
            point_int i = core[c];
            point_int new_root = minE[R[i]].j;
            R[i] = new_root;
            if(new_root == i){
                C.push_back(i);
            }
        }
        
        //check:
        n_tree = C.size();
        dn_tree = n_tree_old - n_tree;
        isteps.push_back(istep);
        n_trees.push_back(n_tree);
        istep += 1;
    }
    return n_tree;


}







