#include <cstdlib>
#include <assert.h>
#include <bits/stdc++.h>
#include <mpi.h>
#include <math.h>
#include <vector>
#include <iostream>
#include <fstream>
#include <cmath>
#include <numeric>     
#include "../include/globals.h"
using namespace std;

void minEdge_vertex(const int rank, const int gsize, const int estart, const int nE, const int n, const int rstart, const int rend, const vector<int> r_counts, int *IA, int *JA, float *A, vector<int> &best, vector<int> &RIA, vector<int> &RJA, vector<int> &min_e, vector<int> &E)
{
    best.resize(n);
    fill(best.begin(), best.end(), -1);
    min_e.resize(n);
    vector<float> minW;
    minW.resize(n);
    int u,v,i;
    float w;

    int n0 = r_counts[0];
    int k, e;
    int n_out = 0;
    RIA.resize(n);
    RJA.resize(nE);

    for (int l = 0; l < nE; l++){
        e = E[l];
        u = IA[e];
        v = JA[e];
        
        RJA[e] = v;
        w = A[e];
        i = u - rstart;
        RIA[i] = u;
    
        if(best[i] < 0){
            best[i] = v;
            min_e[i] = e + estart;
            minW[i] = w;
        }else if(w < minW[i]){
            best[i] = v;
            min_e[i] = e + estart;
            minW[i] = w;
        }
    }
}

void root_outvertices(const int rank, const int gsize, const int rstart, const int rend,const int n0, const vector<int> out_scounts, const vector<int> out_vertices, const vector<int> best, vector<int> &R_new)
{
    int n_out = out_vertices.size();
    vector<int> sbuf(n_out);

     // prepare send
    int* scounts = new int[gsize];
    int* sdispls = new int[gsize];
    int* rcounts = new int[gsize];
    int* rdispls = new int[gsize];     
    for(int i = 0; i<gsize; i++) scounts[i] = out_scounts[i];
    for (int i = 0; i<gsize; i++) sdispls[i] = (i==0) ? 0:sdispls[i-1] + scounts[i-1];

    vector<int> counts(gsize, 0);
    int v, k, pointer;
    for(int i = 0; i<n_out; i++){
        v = out_vertices[i];
        k = floor((float)v/(float)n0);
        pointer = sdispls[k]+counts[k];
        sbuf[pointer] = v;
        counts[k] += 1;
    }

    // send
    MPI_Alltoall(scounts, 1, MPI_INT, rcounts, 1, MPI_INT, MPI_COMM_WORLD);
    k = 0;
    for(int i = 0; i < gsize; i++) {
        rdispls[i] = (i==0) ? 0:rdispls[i-1] + rcounts[i-1];
        k += rcounts[i];
    }
    vector<int> rbuf(k);
    MPI_Alltoallv(&sbuf[0], scounts, sdispls, MPI_INT, &rbuf[0], rcounts, rdispls, MPI_INT, MPI_COMM_WORLD);
    // update
    for(int l = 0; l<k; l++){
    v = rbuf[l];
    rbuf[l] = best[v-rstart];
    }    

   // recv
    vector<int> rbuf2(n_out);
    MPI_Alltoallv(&rbuf[0], rcounts, rdispls, MPI_INT, &rbuf2[0], scounts, sdispls, MPI_INT, MPI_COMM_WORLD);   

   // update
   for(int i = 0; i<n_out; i++){
    v = sbuf[i];
    R_new[v] = rbuf2[i];
   }

   for(int i = rstart; i<rend; i++) R_new[i] = best[i-rstart];

}



void minEdge_node(const int rank, const int gsize, MPI_Datatype Edgetype, const int estart, const int nE, const int n, const int rstart0, const int rstart, const int rend, const vector<int> r_counts, const int n_root, int *IA, float *A, vector<int> &E, int &n_edges, vector<int> &best, const vector<int> R_new, vector<int> &RIA, vector<int> &RJA, vector<int> &min_e)
{

    // 1. find local_min & cutting useless edges;
        //note: only check edges in list E_old, and form a new one for next iteration
        //note: the R is always a function from {1,2,3,...,n_root_old} -> {1,2,3,...,n_root_new}. This is made sure by the relable step in the root finding function.
    int e, u, v, r_u, r_v, i, j;
    float w, minw;
    vector<int> root_index(n_root, -1);
    vector<Edge> minedges(n_root);
    int n_check, ind_root;
    vector<int> E_new;
    int n_edges_new = 0;
    n_check = 0;
    n_edges = E.size();
    for (int l = 0; l<n_edges; l++){
        e = E[l];
        w = A[e];
        u = IA[e] - rstart0;
        i = RIA[u];
        j = R_new[RJA[e]]; //new root

        if (i != j){
            RJA[e] = j;
            E[n_edges_new] = e;
            n_edges_new += 1;

            ind_root = root_index[i];
            if(ind_root <0){
                minedges[n_check] = {i, j, w, e + estart};
                root_index[i] = n_check;
                n_check += 1;
            }else if(w < minedges[ind_root].w){
                    minedges[ind_root] = {i, j, w, e + estart};
            }
        }
    }
    minedges.resize(n_check);
    n_edges = n_edges_new;
    E.resize(n_edges);    
    // 2. all to all transpose
        // prepare send
    int* scount = new int[gsize];
    int* rcount = new int[gsize];
    vector<Edge> sedges;

    int start = 0;
    for (int i = 0; i < gsize; i++){
        scount[i] = 0;

        for (int j = start; j < start + r_counts[i]; j++){
            ind_root = root_index[j];
            if(ind_root > -1){ //there is local min for root j
                sedges.push_back(minedges[ind_root]);
                scount[i]+=1;
            }
        }
        start += r_counts[i];
    }

    if(start != n_root){
        cout<< "r_counts have wrong numbers for some process" << endl;
        exit(100);
    }

    minedges.resize(0);

    MPI_Alltoall(scount, 1, MPI_INT, rcount, 1, MPI_INT, MPI_COMM_WORLD);

        //prepare recv
    int* sdispls = new int[gsize];
    int* rdispls = new int[gsize];
    vector<Edge> redges;
    sdispls[0] = 0;
    rdispls[0] = 0;
    int number = rcount[0];
    for (int i = 1;i<gsize; i++){
        sdispls[i] = sdispls[i-1] + scount[i-1];
        rdispls[i] = rdispls[i-1] + rcount[i-1];
        number += rcount[i];
    }
    redges.resize(number);

        //alltoall
    MPI_Alltoallv(&sedges[0], scount, sdispls, Edgetype, &redges[0], rcount, rdispls, Edgetype, MPI_COMM_WORLD);
    sedges.resize(0);


        // 3. compare and get global min
    best.resize(n);
    fill(best.begin(), best.end(), -1);
 
    vector<float> minW;
    minW.resize(n);
    min_e.resize(n);
    Edge edge;
    int ind;

    for(int l =0; l<number; l++){
        edge = redges[l];
        i = edge.i-rstart;
        if (best[i] <0){
            minW[i] = edge.w;
            best[i] = edge.j;
            min_e[i] = edge.e;
        }else if(edge.w < minW[i]){
            minW[i] = edge.w;
            best[i] = edge.j;
            min_e[i] = edge.e;
        }
    }


}


void form_local_branches(const int rank, const int rstart, const int rend, const int n, vector<int> &best, int &n_local, int &n_root_local, vector<int> &R_local, vector<int> &upper_bound, vector<int> &r_index, vector<int> &local_roots, vector<int> &isroot, vector<int> &r_check2) 
{
    //initilization:
    n_local = 0; // number of local ranches
    n_root_local = 0; // number of local roots 
    R_local.resize(n,-1); // branch labels for each local point
    upper_bound.resize(n); //upper bounds of each branch
    r_index.resize(n); // super label of each branch
    local_roots.resize(n);   //local roots 

    // 1. construct branches by sweeping all local vertices;
    int n_truncate = 0;
    int lj, j, k;


    for(int li = 0; li < n; li++){
        j = best[li];
//        if((rank == 0) and (li == 7287)) cout << rank<<":"<<j<<endl;
        if(j == -2) continue;//non-core is marked as -2 in "best", this only possible in the 1st iteration.
        if(j == -1){//unconnected mst has been touched, this is a root!
            k = li + rstart;
            best[li] = k;
            R_local[li] = n_local;
            upper_bound[n_local] = k;
            r_index[n_local] = n_local;
            n_local += 1;
            local_roots[n_root_local] = k;
            n_root_local += 1;
            continue;
        }
        lj = j - rstart;
        if(R_local[li] < 0){ 
            if(j < rstart || j >= rend){  //li not checked, best connection j outside 
                upper_bound[n_local] = j;
                r_index[n_local] = n_local;
                R_local[li] = n_local;
                n_local += 1;
            } else if(R_local[lj] < 0){ //li not checked, j inside, but j not checked
                R_local[li] = n_local;
                R_local[lj] = n_local;
                upper_bound[n_local] = j;
                r_index[n_local] = n_local;
                n_local += 1;
            } else{ //li not checked, j inside, and j has been checked
                R_local[li] = R_local[lj];
            }
        }else{
            if(j < rstart || j >= rend){ //li checked, best connection j outside
                upper_bound[R_local[li]] = j;
            } else if(R_local[lj] < 0){ //li checked, j inside, but j not checked
                R_local[lj] = R_local[li];
                upper_bound[R_local[li]] = j;
            } else if(R_local[lj] == R_local[li]){ // li checked, j inside and checked, and in same branch: a local root has been found
                // NOTE: THIS IS THE ONLY POSSIBEL CASE SUCH THAT A ROOT AND ITS MINIAL ARE BOTH IN THE range of vertices of current process!
                k = min(li + rstart, j);
                best[li] = k; // confirm a root
                best[lj] = k;
                upper_bound[R_local[li]] = k;
                local_roots[n_root_local] = k;
                n_root_local += 1;
            } else{ // li checked, j inside and checked, but not in same branch: combine these to branch by replacing R_local and r_index in li
                R_local[li] = R_local[lj];
                r_index[R_local[li]] = R_local[lj];
                n_truncate += 1;
            }
        }
    }
    

    // 2. combine branches
    upper_bound.resize(n_local);
    r_index.resize(n_local);
    r_check2.resize(n_local);
    for(int i = 0; i < n_local; i++){
        r_check2[i] = i; //suppose at first every branch needs to be checked
        if(r_index[i] != i){
            r_index[i] = r_index[r_index[i]];
            r_check2[i] = -1; //this branch does not need to be checked since it has been combined to a main branch, we only need to check the upper bound of all main branches.
            while(r_index[r_index[i]] != r_index[i]){
                r_index[i] = r_index[r_index[i]]; // at the same time this root becomes inactive;

            }
        }
    }
 
    // 3. get local roots s.t. both i, j lie in the range of current process
    isroot.resize(n_local, -1); //index for each branch, 1 represents upper bound of this branch now is a root;
    local_roots.resize(n_local); // maximum number of roots should be no more than n_local, this list would be updated after the first iteration to find the roots
    int l;
    for(int i = 0; i< n_root_local; i++){
        l = local_roots[i];
        j = r_index[R_local[l-rstart]]; //local branch index of the root;
        r_check2[j] = -1;
        isroot[j] = 1;
    }

//   for(int i = 0; i< n_local; i++){
//        l = upper_bound[i];
//        j = r_index[i];
//        if((rank == 0) and (r_check2[i] >= 0) and(rstart <= l) and (rend > l)){
//            cout<< l << " " << i << " " << j<< " " <<isroot[j] << " " <<best[l] << " " <<best[best[l]]<<endl;
//        }
//                
//    }

}

void prepare_send(const int istep, const int rank, const int gsize, const int n0, const int n_check, const int n_check2, vector<int> &sbuf, int* sdispls, vector<int> &r_check, const vector<int> rbuf2, const vector<int> r_check2)
{
    vector<int> counts(gsize,0);
    sbuf.resize(n_check);
    r_check.resize(n_check);
    int j, k, pointer;
     if(istep>= 0){
     for(int l = 0; l < n_check2; l++){
        if(r_check2[l] >= 0){
            j = rbuf2[l];
            k = floor((double)j/(double)n0);
            pointer = sdispls[k]+counts[k];
            sbuf[pointer] = j;
            r_check[pointer] = r_check2[l];
            counts[k] += 1;
         }
    }
    }

}


void post_recv(const int rank, const int istep, const int rstart, const int rend, const int n0, const int check0, const int n_local, const int n_check2, int &n_check, const vector<int> sbuf, int **scounts, vector<int> &rbuf2, const vector<int> r_check, vector<int> &r_check2, const vector<int> R_local, vector<int> &upper_bound, vector<int> &r_index, vector<int> &isroot, vector<int> &local_roots, int &n_root_local)
{
    
    int I, J, li, lj, k;
    int combine = 0;
    int *A = NULL;
    A = *scounts;

    for(int l = 0; l < n_check2; l++){
        I = rbuf2[l];
        J = sbuf[l];
        lj = r_check[l];
//    if((rank == 0) and (I==7287)) cout<<"check:"<<l <<" " << J<<endl;   
        if(I < 0 && istep > 0){ // a root confirmed
            r_check2[l] = -1;
            upper_bound[lj] = -I;
            n_check += -1;
            isroot[lj] = 1;
        }else if(rstart <= I && I < rend){
            li = r_index[R_local[I-rstart]];
            if(li == lj){ // root found (which is only possible at the first iteration), the edge associated with the root has been found: I, J
                if(istep >1){
                    cout<<rank << "wrong: " << I<< " " << J <<endl;
                    exit(10);
                }
                k = min(I,J);
                if(I < J){
                    r_check2[l] = -1;
                    upper_bound[lj] = k;
                    n_check += -1;
                    isroot[lj] = 1;
                    local_roots[n_root_local] = I;
                    n_root_local += 1;
                }else{//root found, but index in lower value vertex, next time will return labled root index!
                    k = floor((double)J/(double)n0);
                    A[k] += 1;
                    rbuf2[l] = J;
                }
            }else{ //li != lj, upperbound of li is closer: combine branch lj to branch li
                combine = 1;
                r_index[lj] = li;
                n_check += -1;
                r_check2[l] = -1;
            }
        }else{//otherwise, a closer bound
            k = floor((double)I/(double)n0);
            A[k] += 1;
            upper_bound[lj] = I;
        }
    }
    
    // update r_index: grafting branches
    if(combine == 1){
    for(int i = 0; i < n_local; i++){
        while(r_index[r_index[i]] != r_index[i]){
                r_index[i] = r_index[r_index[i]]; // at the same time upper bound of this local branch becomes inactive;
        }
    }
    }

}

void relable_roots(const int rank, const int gsize, const int n, const int rstart, int &n_root, const int check0, const vector<int> local_roots, const int n_root_local, vector<int> &upper_bound, const int n_local, const vector<int> isroot)
{
    vector<int> R_lable(n,-1);
    int root_start = 0;
    vector<int> n_roots(gsize);
    MPI_Allgather(&n_root_local, 1, MPI_INT, &n_roots[0], 1, MPI_INT, MPI_COMM_WORLD);
    n_root = 0;
    for(int i = 0; i<gsize; i++){
        if(i < rank) root_start += n_roots[i];
        n_root += n_roots[i];
    }

// RELABLE: roots index start from 1, to different the first vertex and first root!
    int j = root_start+1;
    if(rank == 0 && check0 == 1){
        R_lable[0] = 1;
        j = 2;
    
        for(int l = 0; l < n_root_local; l++){
            if(local_roots[l] != 0){
                R_lable[local_roots[l]] = j;
                j += 1;
            }
        }
    }else{
        for(int l = 0; l < n_root_local; l++){
            R_lable[local_roots[l]-rstart] = j;
            j += 1;
        }
    }

   
    for(int i = 0; i < n_local; i++){
    if(isroot[i] >0){
    if(upper_bound[i] < rstart || upper_bound[i]>=rstart + n) cout << "wrong:"<<rank<<"-"<<i<<"->"<<upper_bound[i]<<"-"<<rstart<<"-"<<rstart+n<<endl;
        upper_bound[i] = R_lable[upper_bound[i]-rstart];
    }
    }
}



void count_cycles(const vector<int> I, vector<int> &J, const int n_root, int &n_cycles)
{

    // using the similar way as forming the local branches
    map<int, int> R;
    int k = I.size();

    for(int l =0; l<k; l++){
        R[I[l]] = -1;
    }

    map<int, int> r_index;
    int i,j, r,s;
    int n_local = 0;

    for(int l = 0;l < k; l++){
        i = I[l];
        j = J[l];
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

    n_cycles =0;
    vector<int> local_roots;
    map<int, int> roots;
    for(int l = 0;l<n_local; l++){
        if(r_index[l] == l){
            n_cycles += 1;
            roots[l] = n_root + n_cycles;
        }else{
            while(r_index[r_index[l]] != r_index[l]){
                r_index[l] = r_index[r_index[l]];
            }
        }
    }

    // update
    for(int l = 0; l <k; l++){
        i = I[l];
        j = r_index[R[i]];
        J[l] = roots[j];
    }

}



void break_cycles(const int rank, const int gsize, int &n_root, const int n_check, const int check, const vector<int> checks, const vector<int> r_check2, vector<int> &upper_bound, const vector<int> sbuf, const vector<int> rbuf2)
{
    //1. collect in rank 0, break symmetry and update new roots
    vector<int> I, J;
    if(rank == 0){
        I.resize(check);
        J.resize(check);
    }
    int* rcounts = new int[gsize];
    int* rdispls = new int[gsize];
    if(rank==0){
    for(int i =0;i<gsize;i++){
        rcounts[i] = checks[i];
        rdispls[i] = (i==0) ? 0:rdispls[i-1] + rcounts[i-1];
    }
    }
    MPI_Gatherv(&sbuf[0], n_check, MPI_INT, &I[0], rcounts, rdispls, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Gatherv(&rbuf2[0], n_check, MPI_INT, &J[0], rcounts, rdispls, MPI_INT, 0, MPI_COMM_WORLD);


    int n_cycles;
    if(rank == 0){
        count_cycles(I, J, n_root, n_cycles);
        n_root += n_cycles;
    }
	MPI_Bcast(&n_root, 1, MPI_INT, 0, MPI_COMM_WORLD);
	
    // distribute J
    vector<int> rbuf(n_check);
    MPI_Scatterv(&J[0], rcounts, rdispls, MPI_INT, &rbuf[0], n_check, MPI_INT, 0, MPI_COMM_WORLD);

    //2.update all upper bounds
    int l;
    for(int i=0; i<n_check; i++){
        l = r_check2[i];
        upper_bound[l] = rbuf[i];
//        if (rank ==8) cout<<sbuf[i]<<" " << rbuf2[i] << " " << upper_bound[l]<<endl;
    }
}





int find_root(const int STEP, const int rank, const int gsize, const int n, const int rstart, const int rend, const vector<int> r_counts, const int n_root_old, vector<int> &min_e, vector<int> &minE, vector<int> &best, vector<int> &R_new, const string task)
{
    int n0 = r_counts[0];
    // input: R (R[i]: represents the
    int n_root = 0;
    int log_mode = -2;
    int maxloop = 2*ceil(log2(float(gsize))) + 1;
     
    if(task == "mst") minE.insert(minE.end(), min_e.begin(), min_e.end());


double t0 = MPI_Wtime();
    // 1.form local forests and find roots with edges in this forests
    vector<int> r_check, r_check2;
    vector<int> R_local, upper_bound, r_index, local_roots, isroot;
    int n_local, n_root_local;
    form_local_branches(rank, rstart, rend, n, best, n_local, n_root_local, R_local, upper_bound, r_index, local_roots, isroot, r_check2);

        //CHECK IF 0 IS A ROOT
    int check0=0;

    // 2.locate all roots
    vector<int> sbuf, rbuf, rbuf2;
    int* scounts = new int[gsize];
    int* sdispls = new int[gsize];
    int* rcounts = new int[gsize];
    int* rdispls = new int[gsize];
    for(int i = 0; i<gsize; i++) scounts[i] = 0;
    int n_check = 0;
    int n_check2 = n_local;
    int j, k;
    rbuf2.resize(n_check2);

    for(int i = 0; i < n_local; i++){
        if(r_check2[i] >= 0){
            j = upper_bound[i];
           // if((j<rend) and(j>=rstart)) cout<<rank << " " << j<<endl;
            n_check += 1;
            k = floor((double)j/(double)n0);
            scounts[k] += 1;
            rbuf2[i] = j;
        }else{
               rbuf2[i] = -1;
        }
    }

    vector<int> checks(gsize);
    int istep = 0;
    int check = 1;
    int check_old = 0;
    int lj;
    vector<int> n_roots;
    vector<int> R_lable;
    int root_start, counter;

MPI_Barrier(MPI_COMM_WORLD);
double t1 = MPI_Wtime();
if((rank == 0)  and (log_mode > -1))cout << "time1:" << t1 - t0 << endl;

    while(check > 0){
        //2-1 prepare send
        for (int i = 0; i<gsize; i++) sdispls[i] = (i==0) ? 0:sdispls[i-1] + scounts[i-1];
        prepare_send(istep, rank, gsize, n0, n_check, n_check2, sbuf, sdispls, r_check, rbuf2, r_check2);
         n_check2 = n_check;

if(log_mode > 0) cout << "rank-" << rank <<"-step1" << endl;
        //2-2 send
        MPI_Alltoall(scounts, 1, MPI_INT, rcounts, 1, MPI_INT, MPI_COMM_WORLD);
        k = 0;
        for(int i = 0; i < gsize; i++) {
            rdispls[i] = (i==0) ? 0:rdispls[i-1] + rcounts[i-1];
            k += rcounts[i];
        }
        rbuf.resize(k);
        MPI_Alltoallv(&sbuf[0], scounts, sdispls, MPI_INT, &rbuf[0], rcounts, rdispls, MPI_INT, MPI_COMM_WORLD);


if(log_mode > 0) cout << "rank-" << rank <<"-step2" << endl;
        //2-3 update:
            // 1. if root found: return negative;
            // 2. if not: return a new upper bound (which is closer to the root);
        for(int l = 0; l < k; l++){
            j = rbuf[l]; 
            lj = r_index[R_local[j-rstart]];
            rbuf[l] = upper_bound[lj];

        int jj = isroot[lj];
        if(jj == 1 && istep >0) rbuf[l] = -upper_bound[lj];
     }

if(log_mode > 0) cout << "rank-" << rank <<"-step3" << endl;

        //2-4 recv
        rbuf2.resize(n_check);
        MPI_Alltoallv(&rbuf[0], rcounts, rdispls, MPI_INT, &rbuf2[0], scounts, sdispls, MPI_INT, MPI_COMM_WORLD);

if(log_mode > 0) cout << "rank-" << rank <<"-step4" << endl;

        //2-5 update:
        for(int i = 0; i < gsize; i++) scounts[i] = 0;
        r_check2.resize(n_check);
        r_check2 = r_check;
        post_recv(rank, istep, rstart, rend, n0, check0, n_local, n_check2, n_check, sbuf, &scounts, rbuf2, r_check, r_check2, R_local, upper_bound, r_index, isroot, local_roots, n_root_local);  

if(log_mode > 1) cout << "rank-" << rank <<"-step5-1" << endl;

        //2-5-2
            //for step 0: (1) need to relable roots; (2) need to update check0;
        if(istep == 0){
            relable_roots(rank, gsize, n, rstart, n_root, check0, local_roots, n_root_local, upper_bound, n_local, isroot);   
    }

if(log_mode > 1) cout << "rank-" << rank <<"-step5-2" << endl;

        //2-6 find if n_check == 0 in all process
        MPI_Allgather(&n_check, 1, MPI_INT, &checks[0], 1, MPI_INT, MPI_COMM_WORLD);
        check = 0;
        for (int i = 0; i < gsize; i++) check += checks[i];
  
        //a cyclic chain has been found, need to fix the deadlock formed in approximate graph
        if((check !=0) and (check == check_old)){
            break_cycles(rank, gsize, n_root, n_check,check, checks, r_check2, upper_bound, sbuf, rbuf2);
            check = 0;
        }
        check_old = check;
        MPI_Barrier(MPI_COMM_WORLD);
        istep += 1;

        if(istep > maxloop) {
            //cout<<rank<<"--"<<n_check<<endl;
            MPI_Barrier(MPI_COMM_WORLD);
            exit(12);
        }
    }
MPI_Barrier(MPI_COMM_WORLD);
double t2 = MPI_Wtime();
if((rank == 0) and (log_mode > -1)) cout << "time2:"<<t2 - t1 << endl;

    // 3. pass best to R_new
    int li;
    for(int i = 0; i < n; i++){
        if(best[i] == -2) continue;
        li = r_index[R_local[i]];
        best[i] = upper_bound[li] - 1;
    }
MPI_Barrier(MPI_COMM_WORLD);
double t3 = MPI_Wtime();
if((rank == 0) and (log_mode > -1)) cout << "time3:"<<t3- t2<< endl;

    if(STEP >= 1) R_new.resize(n_root_old);

    for (int i = 0; i < gsize; i++){
        rcounts[i] = r_counts[i];
        if (i != 0) rdispls[i] = rdispls[i-1] + rcounts[i-1];
    }

    MPI_Allgatherv(&best[0], n, MPI_INT, &R_new[0], rcounts, rdispls, MPI_INT, MPI_COMM_WORLD);
MPI_Barrier(MPI_COMM_WORLD);
double t4 = MPI_Wtime();
if((rank == 0)  and (log_mode > -1))cout << "time4:"<<t4- t3 << endl;

    return n_root;
}

void assign_roots(const int rank, const int gsize, const int n_root, int &n, int &rstart, int &rend, vector<int> &r_counts)
{
    int n0, n1, p;
    
    n0 = ceil((double)n_root / (double)gsize);
    p = ceil((double)n_root / (double)n0);
    n1 = n_root - n0*(p-1);
    n = n0;
    rstart = n0 * rank;
    if(rank == p - 1){
       n = n1;
    }else if(rank > p - 1){
       n = 0;
       rstart = n_root;
    }
    rend = rstart + n;

    fill(r_counts.begin(), r_counts.end(), 0);
    for (int i = 0; i< p; i++){
        r_counts[i] = (i < p-1) ? n0 : n1;
    }

}


vector<int> parallel_boruvka(const int N, const int nE, const int estart, int *IA, int *JA, float *A)
{
    int rank, gsize;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &gsize);

    // declare new MPI type: Edgetype
    MPI_Datatype Edgetype;
    MPI_Datatype type[4] = {MPI_INT, MPI_INT, MPI_FLOAT, MPI_INT};
    int blocklen[4] = {1,1,1,1};
    MPI_Aint disp[4];
    disp[0] = offsetof(struct Edge, i);
    disp[1] = offsetof(struct Edge, j);
    disp[2] = offsetof(struct Edge, w);
    disp[3] = offsetof(struct Edge, e);
    MPI_Type_create_struct(4, blocklen, disp, type, &Edgetype);
    MPI_Type_commit(&Edgetype);

    const string task = "mst";
    const string debug = "no";
    // root label stored in RIA
    vector<int> best, R_new, RIA, RJA, E, min_e, minE, r_counts;
    int istep, n_root, n_root_new, dn_root, log_root;
    int n, rstart, rstart0, rend, n_edges, nV, e;

    E.resize(nE);
    for(int i = 0; i < nE; i++) E[i] = i;
    r_counts.resize(gsize);

    istep = 0;
    n_root = N;
    dn_root = 1;
    log_root = 1;

    double t1, t2, time0;

    while(dn_root > 0){

        assign_roots(rank, gsize, n_root, n, rstart, rend, r_counts);
        
        // 1. find_min edges. //input: edge list E, //output: need minE, R
        time0 = MPI_Wtime();
        if(istep == 0){
            minEdge_vertex(rank, gsize, estart, nE, n, rstart, rend, r_counts, IA, JA, A, best, RIA, RJA, min_e, E);
            rstart0 = rstart;
            nV = n;
        }else{
            minEdge_node(rank, gsize, Edgetype, estart, nE, n, rstart0, rstart, rend, r_counts, n_root, IA, A, E, n_edges, best, R_new, RIA, RJA, min_e);
        }
        t1 = MPI_Wtime() - time0;
       // 2. find root
        time0 = MPI_Wtime();
        R_new.resize(0);
        n_root_new = find_root(istep, rank, gsize, n, rstart, rend, r_counts, n_root, min_e, minE, best, R_new, task);
        dn_root = n_root - n_root_new;
        n_root = n_root_new;

        if(log_root == 1){
            for(int i = 0; i<nV; i++){
                if (RIA[i] == -1) continue;
                e = R_new[RIA[i]];
                RIA[i] = e;
            }
        }

        t2 = MPI_Wtime() - time0;

        if(rank == 0) cout << istep << ":" << n_root << "-"<<t1<<"-"<<t2<< endl;
        istep += 1;
    }


    return RIA;
    
}


