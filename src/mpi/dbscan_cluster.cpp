#include <cstdlib>
#include <assert.h>
#include <bits/stdc++.h>
#include <mpi.h>
#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <cmath>
#include <unordered_map>
#include "boruvka.cpp"
#include "../include/globals.h"
using namespace std;


void mark_cores(const int rank, const int gsize, const int N, const int n, const int rstart, int *JA, float *A, vector<int> &best, vector<int> &POINT, vector<int> &R_all, const float eps, const int minPts, int &N_noncore) 
{
    //NOTE: this a fast way marking cores given that the knn graph is from pygofmm output: the neighbors for a point is from nearest to farest;
//modify argus: minPts; delete E, mine; add POINT
    // 1. find local core points: store in RIA
    int minPts0 = minPts - 1; //# edges associated with each point
    vector<int> noncore;
    edge_int pointer;
    for(int i = 0; i< n; i++){
        pointer = (edge_int) (i+1)*minPts0-1;//the edge with largest weight
        if(A[pointer] > eps){ //this is a non core point
            noncore.push_back(i+rstart);
        }    
    }
    int n_noncore = noncore.size();
    // 2. determine global core points: store in core
    int* rcounts = new int[gsize];
    int* rdispls = new int[gsize];
    vector<int> counts(gsize);
    MPI_Allgather(&n_noncore, 1, MPI_INT, &counts[0], 1, MPI_INT, MPI_COMM_WORLD);
    N_noncore = 0;
    rdispls[0] = 0;
    for (int i = 0; i < gsize; i++){
        rcounts[i] = counts[i];
        N_noncore += counts[i];
        if (i != 0) rdispls[i] = rdispls[i-1] + rcounts[i-1];
    }
    vector<int> recv(N_noncore);
    MPI_Allgatherv(&noncore[0], n_noncore, MPI_INT, &recv[0], rcounts, rdispls, MPI_INT, MPI_COMM_WORLD);

    for(int i:recv) R_all[i] = -1; //mark all noncore points as -1
    recv.resize(0);
    noncore.resize(0);
    if(rank == 0) cout << "core:"<<N-N_noncore <<endl;

    // 3. find min_edges
    // NOTE: at this first iteration, we need to :
    // (i) select min-edges s.t. connecting two core ponints;
    // (ii) make sure all edges to be checked afterwards are core-core edges;
    best.resize(n, -1);
    POINT.resize(n, -1);
    int v, counter;
    for(int i = 0; i < n; i++){
        if(R_all[i+rstart] == -1){//noncore point;
            best[i] = -2;
        }else{ //core points: start search min connected core point
            counter = -1;
            pointer = (edge_int) i*minPts0 - 1;
            for(int l =0; l<minPts0; l++){
                pointer += 1;
                counter += 1;
                v = JA[pointer];
                if(R_all[v] != -1){//connected core point found;
                    best[i] = v;
                    break; // a min has been found, break and search for next point;
                }
            }
            POINT[i] = counter;
        }
    }
 
}


void count_dinstinct_roots(const int rank, const int gsize,  const int rstart0, const int nV, const int n_root, const int n0, vector<int> &R, int &n_local, vector<int> &best, vector<int> &s_count)
{
    int rend = rstart0 + nV;
    vector<int> roots(n_root, -1);
    best.resize(nV);
    s_count.resize(gsize, 0);
    int r, k, ind_root;
    n_local = 0;

    for(int i = rstart0; i<rend; i++){
        r = R[i];
        if (r < 0 ) continue; //non-core points
        ind_root = roots[r];
        if(ind_root < 0){
            k = floor((double)r/(double)n0);//bucket index
            best[i-rstart0] = n_local;
            roots[r] = n_local;
            s_count[k] += 1;
            n_local += 1;
        }else{
            best[i-rstart0] = ind_root;
        }
    }
}

void update_vector(const int istart, const int iend, vector<int> &R, const vector<int> R_new){

    int u,r;
    for(int i = istart; i < iend; i++){
        u = R[i];
        if (u < 0 ) continue; //non-core points
        r = R_new[u];
        R[i] = r;
    }
}

void update_roots(const int rank, const int gsize, const int N, const int rstart0, const int nV, const int n_root, const int n0, vector<int> &R, const vector<int> R_new, int &n_local, vector<int> &best, vector<int> &s_count)
{
    int rend0 = rstart0 + nV;
    int u, r, k, ind_root;   
//    update_vector(0, rstart0, R, R_new);

    for(int i = 0; i < rstart0; i++){
        u = R[i];
        if (u < 0 ) continue; //non-core points
        r = R_new[u];
        R[i] = r;
    }


    vector<int> roots(n_root, -1);
    best.resize(nV);
    s_count.resize(gsize);
    fill(s_count.begin(), s_count.end(), 0);
    n_local = 0;

    for(int i = rstart0; i<rend0; i++){
        u = R[i];
        if (u < 0 ) continue; //non-core points
        r = R_new[u];
        R[i] = r;
        ind_root = roots[r];
        if(ind_root < 0){
            k = floor((double)r/(double)n0);//bucket index
            best[i-rstart0] = n_local;
            roots[r] = n_local;
            s_count[k] += 1;
            n_local += 1;
        }else{
            best[i-rstart0] = ind_root;
        }

    }

//    update_vector(rend0, N, R, R_new);
  
    for(int i = rend0; i < N; i++){
        u = R[i];
        if (u < 0 ) continue; //non-core points
        r = R_new[u];
        R[i] = r;
    }  
}


void sweep_cores(const int istep, const int rank, const int gsize, MPI_Datatype Edgetype, const int rstart0, const int nV, const int rstart, const int n, const int n0, const int n_root, const vector<int> s_count, const int n_local, int *JA, float *A, vector<int> &best, vector<int> &POINT, const vector<int> R_all, const int minPts)
{
//rstart0: start index of point stored
//nV: number of points stored
//POINT: swept edge position for each point
//best: local root index of each point(a temporary index: ranges from 0 to n_local -1)

//rstart: start index of roots for this local rank
//n: number of roots for this local rank
//n0: number of roots in rank 0

//n_root: total number of root 
//n_local: local roots for stored points
    int minPts0 = minPts - 1;
    int rend = rstart + n;

    int* scounts = new int[gsize];
    int* rcounts = new int[gsize];
    int* sdispls = new int[gsize];
    int* rdispls = new int[gsize];
    for(int i = 0; i<gsize; i++){
        scounts[i] = s_count[i];
        sdispls[i] = (i == 0) ? 0:sdispls[i-1] + scounts[i-1];
    }
    //1. find local_min: local search does not consider current root (super_vertex) of the connected core point, this will be determined in the global search step
    int i, j, v, counter, position, k, ind_root;
    edge_int pointer;
    vector<int> counts(gsize, 0);
    vector<int> check(n_local, -1);
    vector<newEdge> minedges(n_local);
    int n_check = 0;
    float w;
    for(int u = 0; u < nV; u++){
        v = POINT[u];
        if((0 <= v) and (v < minPts0 - 1)){ //there are rest edges to check
            i = R_all[u+rstart0];
            counter = v;
            pointer = (edge_int) u * minPts0 + v;
            for(int l = v+1; l < minPts0; l++){
                pointer += 1;
                j = R_all[JA[pointer]];
                if((0 <= j) and (i != j)){ //a possible local min found
                    ind_root = best[u]; //local index of root
                    position = check[ind_root]; //local position to sotre this root information
                    w = A[pointer];
                    if(position < 0){// has no position, to create one
                        k = floor((double)i/double(n0));
                        position = sdispls[k] + counts[k];
                        counts[k] += 1;
                        minedges[position] = {i, j, w};
                        check[ind_root] = position;
                    }else if(w < minedges[position].w){
                        minedges[position] = {i, j, w};
                    }
                    break;
                }else{
                    counter += 1;
                }
            }
            POINT[u] = counter;
        }
    }
    check.resize(0);


    //2. transpose edges
    for(int i = 0; i<gsize; i++){
        scounts[i] = counts[i]; //true number of edges to send to rank i;
    } 
    MPI_Alltoall(scounts, 1, MPI_INT, rcounts, 1, MPI_INT, MPI_COMM_WORLD);
    int number = 0;
    for(int i = 0; i<gsize; i++){
        rdispls[i] = (i ==0) ? 0: rdispls[i-1] + rcounts[i-1];
        number += rcounts[i];
    }

    vector<newEdge> redges(number);
    MPI_Alltoallv(&minedges[0], scounts, sdispls, Edgetype, &redges[0], rcounts, rdispls, Edgetype, MPI_COMM_WORLD);
    minedges.resize(0);

    // 3. compare and get global min
    best.resize(n); // after 1st iteration, best can only be -1 (self root, maybe connected by another vertex) or non-negative 
    fill(best.begin(), best.end(), -1);
    vector<float> minW(n);
    newEdge edge;
    for(int l =0; l<number; l++){
        edge = redges[l];
        i = edge.i-rstart;
        if (best[i] <0){
            minW[i] = edge.w;
            best[i] = edge.j;
        }else if(edge.w < minW[i]){
            minW[i] = edge.w;
            best[i] = edge.j;
        }
    }
}



void label_borders(const int nV, const int rstart0, vector<int> &R_all, int *JA, float *A, const float eps, const int minPts, int &n_border)
{
// label border points: by assigning cluster label of its nearest core point
// k: number of non zero weights for each point
// eps: radius of consideration
    int if_border, j, minj;
    edge_int pointer;   
    int minPts0 = minPts - 1; 
    float w;
    vector<Border> borders;
    for(int i = 0; i < nV; i++){
        if(R_all[i+rstart0] < 0){ //if a non-core point
            if_border = 0;  //assume as noise at first
            pointer = (edge_int) i*minPts0 -1;
            for(int l = 0; l < minPts0; l++){
                pointer += 1;
                w = A[pointer];
                j = R_all[JA[pointer]];
                if((j >= 0) and (w < eps)){ //connected to a core within the radius of eps around point i: this is a border point.
                    borders.push_back({i+rstart0, j});
                    break;
                }
            }
        }
    }    

    n_border = borders.size();
    Border a;
    for(int i = 0; i<n_border; i++){
        a = borders[i];
        R_all[a.b] = a.label;
    }
}


void output(vector<int> R, string name)
{
    int n = R.size();
    ofstream fout;
    fout.open(name);
    for(int i = 0; i<n; i++){
        fout<<R[i]<<"\n";
    }

}

vector<int> compare(vector<int> R1, int n, string name, int mode)
{
    int rank, gsize;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &gsize);
cout<<"start:"<<rank<<" :" <<name<<endl;
    vector<int> R2(n);
    ifstream fin(name);
    for(int i=0; i<n; i++){
        fin>>R2[i];
    }
    fin.close();

    if(mode == 1){
    for(int i = 0; i<n; i++){
        if(R1[i] != R2[i]) cout<< i <<":"<<R1[i]<<" " <<R2[i]<<endl;
    }
    }

    if(mode == 2){
    for(int i = 0; i<n; i++){
        if((R1[i] != R2[i]) and (R1[i] >= 0)) cout<< i <<":"<<R1[i]<<" " <<R2[i]<<endl;

        if(rank == 0) cout<< i<<endl;
    }
    }
cout<<"end:"<<rank<<" :"<<name<<endl;
}

vector<int> DBSCAN_MST(const int N, int *JA, float *A, const float eps, const int minPts)
{
    int rank, gsize;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &gsize);

    // declare new MPI type: Edgetype
    MPI_Datatype Edgetype;
    MPI_Datatype type[3] = {MPI_INT, MPI_INT, MPI_FLOAT};
    int blocklen[3] = {1,1,1};
    MPI_Aint disp[3];
    disp[0] = offsetof(struct newEdge, i);
    disp[1] = offsetof(struct newEdge, j);
    disp[2] = offsetof(struct newEdge, w);
    MPI_Type_create_struct(3, blocklen, disp, type, &Edgetype);
    MPI_Type_commit(&Edgetype);


    // get clusters for core points: labels are stored in RIA
    const string task = "dbscan";
    const string debug = "no";
    vector<int> best, POINT, R_new, min_e, minE, r_counts, out_vertices, out_scounts;
    int istep, n_root, n_root_new, dn_root, log_root;
    int n, rstart, rstart0, rend, rend0, n_edges, nV, nV0, e;
    vector<int> R_all(N), s_count(gsize);
    int n_local, n0;

    r_counts.resize(gsize);
    out_vertices.resize(0);
    out_scounts.resize(0);

    istep = 0;
    n_root = N;
    dn_root = 1;
    log_root = 1;

    int n_border, N_border, N_noncore;
string name1;
    double t1, t2, time0;
    double total_t0 = MPI_Wtime();
    while(dn_root > 0){

        assign_roots(rank, gsize, n_root, n, rstart, rend, r_counts);
        time0 = MPI_Wtime();

        if(istep == 0){
            // 1.1: mark core points and find min: first step local min is global min
            mark_cores(rank, gsize, N, n, rstart, JA, A, best, POINT, R_all, eps, minPts, N_noncore);
            rstart0 = rstart;
            rend0 = rstart0 + n;
            nV = n;
            nV0 = r_counts[0];
            t1 = MPI_Wtime() - time0;

            // 1.2: pointer jumping to find the root: use R_all at first itr
            n_root_new = find_root(istep, rank, gsize, n, rstart, rend, r_counts, n_root, min_e, minE, best, R_all, task);
        }else{
            //2.0: update roots
            n0  = r_counts[0];
            if(istep == 1){
                count_dinstinct_roots(rank, gsize, rstart0, nV, n_root, n0, R_all, n_local, best, s_count);
            }else{
                update_roots(rank, gsize, N, rstart0, nV, n_root, n0, R_all, R_new, n_local, best, s_count);
            }

            // 2.1: sweep core points to find global min
            sweep_cores(istep, rank, gsize, Edgetype, rstart0, nV, rstart, n, n0, n_root, s_count, n_local, JA, A, best, POINT, R_all, minPts);
            t1 = MPI_Wtime() - time0;

            // 2.2: pointer jumping to find the root: use R_new
            n_root_new = find_root(istep, rank, gsize, n, rstart, rend, r_counts, n_root, min_e, minE, best, R_new, task);
        }

        dn_root = n_root - n_root_new;
        n_root = n_root_new;
        t2 = MPI_Wtime() - time0;

        if(rank == 0) cout << istep << ":" << n_root << "-"<<t1<<"-"<<t2<< endl;

        istep += 1;
    }

    update_vector(0, N, R_all, R_new);

    // get clusters for border points;
    label_borders(nV, rstart0, R_all, JA, A, eps, minPts, n_border);
    vector<int> R_local(R_all.begin()+rstart0, R_all.begin()+rend0);

    double total_tend = MPI_Wtime(); 
 
    vector<int> counts(gsize);
    N_border = 0;
    MPI_Allgather(&n_border, 1, MPI_INT, &counts[0], 1, MPI_INT, MPI_COMM_WORLD);
    for(int i =0;i<gsize; i++) N_border += counts[i];

    if(rank == 0) {
        cout<<"DBSCAN_clustering time:"<<total_tend - total_t0<<endl;
        cout<<"number of core points:"<<N-N_noncore<<endl;
        cout<<"number of border points:"<<N_border<<endl;
        cout<<"number of noise points:"<<N_noncore - N_border<<endl;
    }

    for(int i =0; i<nV; i++){
        if(R_local[i] < 0) R_local[i] = -1;
    }
    
    return R_local;

} 

void count_vertices(const int rank, const int gsize, const int n_root, int &n, int &rstart, int &rend, vector<int> &r_counts)
{
    //count number of vertices at each iteration for each process used in boruvka's algo
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

void count_points(const int rank, const int gsize, const int N, int &nV0, int &istart0, vector<int> &counts)
{
    //count number of points for each process used in pyGofmm
    int n0, pstart;
    n0 = floor((double)N/(double)gsize);
    pstart = N%gsize;
    nV0 = n0;
    if(rank < pstart){
        nV0 = n0+1;
        istart0 =(n0+1)*rank;
    }else{
        istart0 = (n0+1) * pstart + n0 * (rank - pstart);
    }

    counts.resize(gsize);
    for (int i=0; i<gsize; i++){
        if(i<pstart){
            counts[i] = n0+1;
        }else{
            counts[i] = n0;
        }
    }
}

void cyclic_to_seq(const int rank, const int gsize, const int N, vector<int> &R)
{
    //reindex gids element from cyclic indexing to sequential one
    // NOTE: this exactly depends on the number of processes
    int nV, istart;
    vector<int> counts, starts;
    count_points(rank, gsize, N, nV, istart, counts);
    
    starts.resize(gsize);
    int k = 0;
    for(int i =0; i<gsize; i++){
        starts[i] = k;
        k += counts[i];
    }
    
    int j, index;
    k = 0;
    edge_int n = R.size();
    for(edge_int i = 0; i < n; i++){
        index = R[i];
        j = starts[index%gsize];
        k = floor((double)index/(double)gsize);
        R[i] = j + k;
    }

}


vector<int> seq_to_cyclic(const int rank, const int gsize, const int N, vector<int> &R)
{
    int nV, istart;
    vector<int> counts, starts;
    count_points(rank, gsize, N, nV, istart, counts);

    starts.resize(gsize);
    int k = 0;
    for(int i =0; i<gsize; i++){
        starts[i] = k;
        k += counts[i];
    }

    int j;
    k = 0;
    vector<int> labels(N);
    for(int i =0; i<N;i++){
        j = starts[i%gsize];
        k = floor((double)i/(double)gsize);
        labels[i] = R[j+k];
    }
    R.resize(0);

    return labels;
}




void redistribute_matrix(const int rank, const int gsize, int **pJA, float **pA, const vector<int> gids, const vector<float> distances, const int N, const int k)
{
//k = minPts-1
    int *JA = NULL;
    float *A = NULL;

    //count number
    int n0, istart0, pstart, nV0; //knn uses
    int n, n1, p, nV, istart;    //mst uses
    n = ceil((double)N/(double)gsize);
    p = ceil((double)N/(double)n);
    n1 = N - n*(p-1);
    nV = n;
    istart = n * rank;
    if(rank == p-1){
        nV = n1;
    }else if(rank > p-1){
        nV = 0;
        istart = N;
    }

    n0 = floor((double)N/(double)gsize);
    pstart = N%gsize;
    nV0 = n0;
    if(rank < pstart){
        nV0 = n0+1;
        istart0 =(n0+1)*rank;
    }else{
        istart0 = (n0+1) * pstart + n0 * (rank - pstart);
    }

    edge_int nE = (edge_int) nV * k; //number of edges for local process
    *pJA = JA = (int*)malloc(nE*sizeof(int));
    *pA = A = (float*)malloc(nE*sizeof(float));
//cout<<rank<<"number_0:"<<istart0 << " " <<istart <<endl;
//cout<<rank <<"number:"<<n<< " " <<nV << " " <<p << endl;
    // declare new MPI type: Edgetype
    MPI_Datatype Edgetype2;
    MPI_Datatype type[2] = {MPI_INT, MPI_FLOAT};
    int blocklen[2] = {1,1};
    MPI_Aint disp[2];
    disp[0] = offsetof(struct EdgeSend, j);
    disp[1] = offsetof(struct EdgeSend, w);
    MPI_Type_create_struct(1, blocklen, disp, type, &Edgetype2);
    MPI_Type_commit(&Edgetype2);

    //allocate local edges
    int pointer0 = 0;
    int dp, n_send, n_recv;
    if((pstart != 0) and (pstart != gsize - 1) and (rank >= pstart)){
        // prepare send recv count
        dp = rank - pstart;
        n_send = dp;
        n_recv = dp + 1;
        if(dp>n0) n_send = n0;
        if(dp+1>n0) n_recv = n0;
        if(rank == gsize-1) n_recv = 0;
        pointer0 = n_send;
    }
    edge_int pointer = 0;
    int ia, point0;
    edge_int inst;
    for(int m= pointer0; m < nV0; m++){
        ia = istart0 + m;
        for(int l = 0; l < k;l++){
            inst = (edge_int) (l+1)*nV0 + m;
            if(gids[inst] == ia){
                 inst = m;
            }
            *(JA+pointer) = gids[inst];
            *(A+pointer) =  sqrt(distances[inst]);
            pointer += 1;
        }
    }

    // allocate all other exchanged edges
    if((pstart != 0) and (pstart != gsize - 1) and (rank >= pstart)){
        //send
        if(rank > pstart){
            vector<EdgeSend> sbuf(n_send*k);
            int count = 0;
            for(int i = 0; i < n_send; i++){
                point0 = istart0 + i;
                for(int l = 0; l < k; l++){
                    inst = (l+1) * nV0 + i;
                    if(gids[inst] == point0) inst = i;
                    sbuf[count] = {gids[inst], sqrt(distances[inst])};
                    count += 1;
                }
            }

            MPI_Send(&sbuf[0], n_send*k, Edgetype2, rank - 1,0,MPI_COMM_WORLD);
        }


        //recv and add new edges to IA, JA, and A
        if((rank >= pstart) and (rank < gsize-1)){
            vector<EdgeSend> rbuf(n_recv*k);
            MPI_Recv(&rbuf[0], n_recv * k, Edgetype2, rank + 1, 0, MPI_COMM_WORLD,MPI_STATUS_IGNORE);
            int count = 0;
            for(int i = 0; i<n_recv; i++){
                ia = ia + i + 1;
                for(int l = 0; l < k; l++){
                    *(JA+pointer) = rbuf[count].j;
                    *(A+pointer) = rbuf[count].w;
                    pointer += 1;
                    count += 1;
                }
            }

        }
    }


}



