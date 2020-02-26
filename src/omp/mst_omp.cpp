#include <cstdlib>
#include <iostream>
#include <fstream>
#include <omp.h>
#include <assert.h>
#include <vector>
#include <string>
#include <cmath>
#include <math.h>
#include <bits/stdc++.h>
#include <time.h>
#include <algorithm>
#include "../include/globals.h"
using namespace std;



bool cas(Edge &e_old, const float w_old, const point_int j_new, const float w_new)
{
    if(e_old.w == w_old){
        e_old = {j_new, w_new};
        return true; //swap success, and terminate write
    }else{
        return false; //swap failed, need to check if it is still necessary for swap
    }

}


void pwrite(Edge &e_old, const point_int j_new, const float w_new)
{
    // compare and swap smaller edges
    float w_old;
    do {
        w_old = e_old.w;
    }
    while ((w_new < w_old) and (cas(e_old, w_old, j_new, w_new)));
    // the first condition determines if it is necessary to swap
    // the second condition determines if the swap is successed.
}

void pointer_jumping(const point_int i, Edge &e1, const point_int j, vector<point_int> &cycleFlags, point_int &count)
{
    if(e1.j != j){
        e1.j = j;
    }else if(cycleFlags[i] >= 0){
        cycleFlags[i] = -1;
        count += 1;
    }

}


void break_cycles(const vector<Cycle> cycles, vector<Edge> &minE)
{

    // using the similar way as forming the local branches
    map<point_int, point_int> R;
    int k = cycles.size();

    point_int i, j, r, s;

    for(int l =0; l<k; l++){
        i = cycles[k].i;
        R[i] = -1;
    }

    map<int, int> r_index;
    int n_local = 0;

    for(int l = 0; l < k; l++){
        i = cycles[l].i;
        j = cycles[l].j;
        if(R[i] < 0){
            if(R[j] < 0){
                R[i] = n_local;
                R[j] = n_local;
                r_index[n_local] = n_local;
                n_local += 1;
            }else{
                R[i] = R[j];
            }
        } else{
            if(R[j] < 0){
                R[j] = R[i];
            } else{
                r = R[i];
                s = R[j];
                if(r_index[r] != r_index[s]){
                    r_index[r] = r_index[s];
                    R[i] = s;

                }
            }
        }

    }


    int n_cycles =0;
    map<int, point_int> roots;
    for(int l = 0;l<n_local; l++){
        if(r_index[l] == l){
            n_cycles += 1;
            roots[l] = -1;
        }else{
            while(r_index[r_index[l]] != r_index[l]){
                r_index[l] = r_index[r_index[l]];
            }
        }
    }
    cout<< "number of points and cycles:"<<k << " " <<n_cycles<<endl;

    // update roots
    int c;
    for(int l = 0; l <k; l++){
        i = cycles[l].i;
        c = r_index[R[i]];
        if(roots[c] == -1){
            roots[c] = i;
            minE[i].j = i;
        }else{
            minE[i].j = roots[c];
        }
    }

}

vector<point_int> parallel_mst(const point_int N, const edge_int *IA, const point_int *JA, const float *A)
{
    int num_threads;
    #pragma omp parallel
    {
        num_threads = omp_get_num_threads();
    }
    cout<< "number of threads:"<< num_threads<<endl;
    float sentinel = 100.;
    point_int n_tree = N;
    point_int dn_tree = 1;

    vector<point_int> R(N), checked(N), cycleFlags(N);
    vector<Edge> minE(N);
    vector<point_int> C;
    vector<Cycle> cycles;

    point_int count;
    vector<point_int> counts(num_threads);
//for debug
vector<int> n_tree_threads(num_threads);
    int istep = 0;
    point_int n_tree_old;
    #pragma omp declare reduction (merge : vector<point_int> : omp_out.insert(omp_out.end(), omp_in.begin(), omp_in.end()))
    #pragma omp declare reduction (merge2 : vector<Cycle> : omp_out.insert(omp_out.end(), omp_in.begin(), omp_in.end()))


    cout<<"   step  " << "|| n_tree " << "||  -I-   ||" << "   -II-  ||" <<"  -III-  ||"<<"  -IV-   ||"<<"   -V-   ||"<<"   -VI-  ||"<<endl;
    vector<double> phase_time(7), dt(7);
    double t0, t1, t2, t3, t4,t5,t6;

    while(dn_tree > 0){

        //phase-I: initialization
        t0 = omp_get_wtime();
        if(istep != 0){
            #pragma omp parallel for
            for(point_int c = 0; c < n_tree; c++)
            {
               point_int i = C[c];
               minE[i] = {-1, sentinel};
            }
        }else{
            #pragma omp parallel for
            for(point_int p = 0; p < N; p++)
            {
                R[p] = p;
                cycleFlags[p] = p;
                minE[p] = {-1, sentinel};
            }
        } 
        t1 = omp_get_wtime();
        dt[1] = t1 - t0;
 
        //phase-II: find min_edges
        #pragma omp parallel for 
        for(point_int p = 0; p < N; p++)
        {
            edge_int m = IA[p] + checked[p];
            while(m < IA[p+1]){
                point_int q = JA[m];
                if(R[p] != R[q]){
                    pwrite(minE[R[p]], R[q], A[m]);
                    break;
                }else{
                    checked[p] += 1;
                    m += 1;
                }
            } 
        } 
        t2 = omp_get_wtime();
        dt[2]= t2 - t1;

        //phase-III: break symmetry
        if(istep == 0){
            #pragma omp parallel for 
            for(point_int p = 0; p < N; p++)
            {
                point_int j = minE[p].j;
                if(j == -1){
                    minE[p].j == p;
                }else{
                    point_int k = minE[j].j;
                    if((p == k) and (p < j)){
                         minE[p].j = p;
                    }
                }
            }
        }else{
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
        }
        t3 = omp_get_wtime();
        dt[3] = t3 - t2;

        //phase-IV: pointer jumping
        if(istep != 0){
            #pragma omp parallel for
            for(point_int c = 0; c < n_tree; c++)
            {
                point_int i = C[c];
                cycleFlags[i] = minE[i].j;
            }
        }    
        count = 1;
        while(count > 0) {
            count = 0;
            if(istep == 0){
               #pragma omp parallel for 
                for(point_int p = 0; p < N; p++)
                {
                    point_int j = minE[minE[p].j].j;
                    pointer_jumping(p, minE[p], j, cycleFlags, counts[omp_get_thread_num()]);
                }
            }else{
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
            }
            for(int t = 0; t<num_threads; t++) count += counts[t]; 
            fill(counts.begin(), counts.end(), 0);
        }
        t4 = omp_get_wtime();
        dt[4] = t4 - t3;


        //phase-V: break cycles
        cycles.resize(0);
        if(istep == 0){
            #pragma omp parallel for reduction(merge2: cycles)
            for(point_int p = 0; p < N; p++)
            {
                if(cycleFlags[p] != -1){        
                    cycles.push_back({p, cycleFlags[p]});
                }
            }

        }else{
            #pragma omp parallel for reduction(merge2: cycles)
            for(point_int c = 0; c<n_tree; c++)
            {
                point_int i = C[c];
                if(cycleFlags[i]!=-1){
                    cycles.push_back({i, cycleFlags[i]});
                }
            }

        }
        if(cycles.size() > 0){
            break_cycles(cycles, minE);
        }
        t5 =  omp_get_wtime();
        dt[5] = t5 - t4;

        //phase-VI: update roots R, number of trees and clusters
             
        n_tree_old = n_tree;
        n_tree = 0;
        C.resize(0);
        fill(n_tree_threads.begin(), n_tree_threads.end(), 0);
        #pragma omp parallel for reduction(merge: C)
        for(point_int p = 0; p < N; p++)
        {
            point_int new_root = minE[R[p]].j;
            R[p] = new_root; 
            if(new_root == p){
                C.push_back(p);
                n_tree_threads[omp_get_thread_num()] += 1;
            }
        }
        t6 = omp_get_wtime();
        dt[6] = t6 - t5;
    
        //check:
        n_tree = C.size();
        int tcount = 0;
        for(int j = 0; j < num_threads; j++) tcount += n_tree_threads[j];
        if(n_tree != tcount){
            cout<<"wrong tree number" << " " << istep << endl;
            exit(12);
        }
        dn_tree = n_tree_old - n_tree;

        for(int c = 1; c<7; c++) phase_time[c] += dt[c];   
        printf("step  %2d :(%8d)   %.2E   %.2E   %.2E   %.2E   %.2E   %.2E\n",istep, n_tree_old, dt[1], dt[2], dt[3], dt[4], dt[5], dt[6]); 
        istep += 1;
    }

        printf("total %2d :(%8d)   %.2E   %.2E   %.2E   %.2E   %.2E   %.2E\n",istep-1, n_tree, phase_time[1], phase_time[2], phase_time[3], phase_time[4], phase_time[5], phase_time[6]);
    double total_time = 0.;
    for(int c = 1; c< 7; c++) total_time += phase_time[c];
    cout<<"total_time:"<<total_time<<endl;
    return R;
}

