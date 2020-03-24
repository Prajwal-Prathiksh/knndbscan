#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <time.h>
#include <mpi.h>
#include <math.h>
#include <fstream>
#include <algorithm>
#include <iostream>
#include <cstdlib>
#include <vector>
#include "dbscan_cluster.cpp"
#include "../include/dbscan.h"
#include "../include/globals.h"
using namespace std;

//redistribute points to each process in cyclic ordering for pygofmm
vector<float> redistribute_points(const int N, const int d, const int C, const int istart, const int npoints, vector<float> point_set)
{
    int rank, gsize;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &gsize);
        
//    for(int i = 0;i<20; i++) cout<<point_set[i]<<endl;

    int* scounts = new int[gsize];
    int* sdispls = new int[gsize];
    int* rcounts = new int[gsize];
    int* rdispls = new int[gsize];
    for(int i = 0;i<gsize;i++) scounts[i] = 0;
    //check points dimension
    int send_size = point_set.size();
    if(send_size != npoints*d){
        cout<<"wrong number of points"<<endl;
        exit(3);
    }

    //determine basic counting parameters
    int pointer;
    for(int i = 0; i <npoints; i++){
        pointer = (i+istart)%gsize;
        scounts[pointer] += d;
    }
    for(int i =0; i<gsize;i++) sdispls[i] = (i==0) ? 0:sdispls[i-1] + scounts[i-1];

    //allocate send and recv information
    vector<float> sbuf(send_size);
    vector<int> counts(gsize, 0);
    int k;
    int index =0;
    for(int i =0; i<npoints;i++){
        k = (i+istart)%gsize;
        pointer = sdispls[k] + counts[k];
        for(int j = 0; j<d; j++) sbuf[pointer + j] = point_set[index+j];
        index += d;
        counts[k] += d;
    }
    point_set.resize(0);
    vector<float> rbuf;

    MPI_Alltoall(scounts, 1, MPI_INT, rcounts, 1, MPI_INT, MPI_COMM_WORLD);
    k = 0;
    for(int i = 0; i < gsize; i++) {
        rdispls[i] = (i==0) ? 0:rdispls[i-1] + rcounts[i-1];
        k += rcounts[i];
    }
    rbuf.resize(k);

    int nV, n0, pstart;
    n0 = floor((float)N/(float)gsize);
    pstart = N%gsize;
    nV = n0;
    if(rank < pstart) nV += 1;
    if(k != nV*d){
        cout<<"wrong send and recv number"<<endl;
        exit(31);
    }

    //redistribute points
    MPI_Alltoallv(&sbuf[0], scounts, sdispls, MPI_FLOAT, &rbuf[0], rcounts, rdispls, MPI_FLOAT, MPI_COMM_WORLD);

    //reordering
    sbuf.resize(0);
    index = 0;
    vector<float> points(k);
    for(int i = 0; i<nV; i++){
        for(int j = 0; j<d; j++){
            pointer = i + j * nV;
            points[pointer] = rbuf[index];
            index += 1;
        }
    }

    return points;
}


vector<int> kNN_DBSCAN(const int N, const float eps, const int minPts, vector<int> gids, vector<float> distances)
{
//parameters:
//    N: problem size, number of points for clustering;
//    eps: radius of consideration in knn_dbscan (density parameter);
//    minPts: minmum points for density (density parameter);
//    gids: size of (minPts*nV), nearest neighbor list, nV is the number of points for local process in pygofmm
//    distances: weights of all nearest neighbors
 

    int rank, gsize;
    int k = minPts-1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &gsize);

    int *JA = NULL;
    float *A = NULL;
    //reordering gids elements
    cyclic_to_seq(rank, gsize, N, gids);

    //redistribute matrix in boruvka setting
    redistribute_matrix(rank, gsize, &JA, &A, gids, distances, N, k);
    if(rank == 0) cout <<"number of processors: " << gsize << endl;
    gids.resize(0);
    distances.resize(0);
    MPI_Barrier(MPI_COMM_WORLD);

    //boruvka iteration to get clustering results
    vector<int> R;
    R = DBSCAN_MST(N, JA, A, eps, minPts);
    free(JA);
    free(A);

    //collect all labels in rank0
    if(rank == 0) cout<< "finish boruvka iterations" << endl;
    int nV, rend, rstart;
    vector<int> r_counts(gsize);
    int* rcounts = new int[gsize];
    int* rdispls = new int[gsize];
    count_vertices(rank, gsize, N, nV, rstart, rend, r_counts);
    rdispls[0] = 0;
    for (int i = 0; i < gsize; i++){
        rcounts[i] = r_counts[i];
        if (i != 0) rdispls[i] = rdispls[i-1] + rcounts[i-1];
    }
    vector<int> rbuf, labels;
    if(rank == 0) rbuf.resize(N);
    MPI_Gatherv(&R[0], nV, MPI_INT, &rbuf[0], rcounts, rdispls, MPI_INT, 0,  MPI_COMM_WORLD);

    //reordering labels in rank0
    if(rank==0){
        labels = seq_to_cyclic(rank, gsize, N, rbuf);
    }
    MPI_Finalize();
    return labels;

}

vector<int> kNNgraph_DBSCAN(const int N, const float eps, const int minPts, const int graph_k, const int nfiles, const string name)
{
    //read graph directly
    int rank, gsize;
    int k = minPts-1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &gsize);    
    
    //number of points in each file;    
    vector<int> npoints_file(nfiles), istart_file(nfiles), iend_file(nfiles);
    int n0_file = floor((double) N/(double) nfiles); //number of points in each of first (nfiles -1) files;
    int n1_file = N - (nfiles - 1) * n0_file; // number of  points in the last file

    for(int i =0; i<nfiles; i++){
        npoints_file[i] = (i == nfiles-1) ? n1_file:n0_file;
        istart_file[i] = i * n0_file;
        iend_file[i] = istart_file[i] + npoints_file[i];    
    }

    //number of points in each process
    int nV, iend, istart;
    vector<int> r_counts(gsize);
    count_vertices(rank, gsize, N, nV, istart, iend, r_counts);
    edge_int nE = nV *k;

    int *JA = (int*)malloc(nE*sizeof(int));
    float *A = (float*)malloc(nE*sizeof(float));

    int nfile_read, read_start, read_end, line_skip, line_read;
    int count0 =0;
    for(int i =0; i <nfiles; i++){
        if((count0 == 0) and (istart_file[i] <= istart)){
            read_start = i;
            count0 = 0;
            line_skip = istart - istart_file[i];
        }
        if((iend > istart_file[i]) and (iend <= iend_file[i])){
            read_end = i;
            line_read = iend - istart_file[i];
        }
        
    }


    nfile_read = read_end - read_start + 1;
//    cout<<rank<<"tt"<< nfile_read<<"+" << read_start<<"+"<<read_end<<"+"<<line_skip<<"+"<<line_read<<endl;
    //read gids
    int nskip, nread, nouse, file_NO;
    string line, filename;
    int index = istart;
    int counter = 0;
    for(int i = 0; i<nfile_read; i++){
        file_NO = read_start + i;
        filename = name + "_gids_" + to_string(file_NO) + ".txt";
        ifstream fin(filename);
        nskip = (i == 0) ? line_skip:0; // only the first file needs to skip the first few line;
        nread = (i == nfile_read-1) ? line_read: npoints_file[file_NO]-nskip;
// number of points to read

        for(int j = 0; j < nskip; j++) getline(fin, line);
        for(int j = 0; j < nread; j++){
            fin >> nouse;
            for(int m = 0; m < graph_k-1; m++){
                if(m<k){
                fin >> JA[counter];
                counter += 1;
                }else{
                fin>>nouse;
                }
            }
            index += 1;
        }
        fin.close();
    }


    //read distances
    float data;
    counter = 0;
    for(int i = 0; i<nfile_read; i++){
        file_NO = read_start + i;
        filename = name + "_distances_" + to_string(file_NO) + ".txt";
        ifstream fin(filename);
        nskip = (i == 0) ? line_skip:0; // only the first file needs to skip the first few line;
        nread = (i == nfile_read-1) ? line_read: npoints_file[file_NO]-nskip;
// number of points to read

        for(int j = 0; j < nskip; j++) getline(fin, line);
        for(int j = 0; j < nread; j++){
            fin >> data;
            for(int m = 0; m < graph_k-1; m++){
                if(m<k){
                fin >> A[counter];
                counter += 1;
                }else{
                fin>>data;
                }
            }
        }
        fin.close();
    }
   
 
    //run dbscan
    MPI_Barrier(MPI_COMM_WORLD);
    vector<int> R;
    R = DBSCAN_MST(N, JA, A, eps, minPts);

    //collect all labels in rank0
    if(rank == 0) cout<< "finish boruvka iterations" << endl;
    int* rcounts = new int[gsize];
    int* rdispls = new int[gsize];
    rdispls[0] = 0;
    for (int i = 0; i < gsize; i++){
        rcounts[i] = r_counts[i];
        if (i != 0) rdispls[i] = rdispls[i-1] + rcounts[i-1];
    }
    vector<int> labels;
    if(rank == 0) labels.resize(N);
    MPI_Gatherv(&R[0], nV, MPI_INT, &labels[0], rcounts, rdispls, MPI_INT, 0,  MPI_COMM_WORLD);

    MPI_Finalize();

    return labels;


}

void output_graph_multi(const int n, const int k, const vector<int> gids, const vector<float> distances, const string name)
{
    int rank, gsize;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &gsize);
    string gids_name = name + "_gids_" + to_string(rank) + ".txt";
    string dist_name = name + "_distances_"+to_string(rank) + ".txt";
    ofstream fout;

    fout.open(gids_name);
    edge_int pointer = 0;
    for(int i =0; i<n;i++){
        for(int j =0; j<k; j++){
            pointer = i + j *n;
            fout<<gids[pointer] <<" " ;
        }
        fout<<"\n";
    }
    fout.close();

    fout.open(dist_name);
    pointer = 0;
    for(int i =0; i<n;i++){
        for(int j =0; j<k; j++){
            pointer = i + j*n;
            fout<<sqrt(distances[pointer]) <<" " ;
        }
        fout<<"\n";
    }
    fout.close();

}



// output knn from 1 process to multiple files
void output_graph(const int N, const int k, const int nfiles, const vector<int> gids, const vector<float> distances, const string name)
{    
    
    int n0 = floor((double) N/(double) nfiles); //number of points in each of first (nfiles -1) files;
    int n1 = N - (nfiles - 1) * n0; // number of  points in the last file
    string filename;
    ofstream fout;
    int n = n0;
    int counter, pointer, istart, iend;

    for (int i = 0; i < nfiles; i++){
        filename = name + "_gids_" + to_string(i) + ".txt";
        fout.open(filename);
        if(i == nfiles-1) n = n1;
        istart = n0*i;
        iend = istart +n;
        for(int j = istart; j <iend; j++){
            for(int m = 0; m < k; m++){
                pointer = m*N + j;
                if(gids[pointer] == j) pointer = j;
                fout << gids[pointer] << " ";
            }
            fout<<"\n";
        }
        fout.close();
    }
    
    for (int i = 0; i < nfiles; i++){
        filename = name + "_distances_" + to_string(i) + ".txt";
        fout.open(filename);
        if(i == nfiles-1) n = n1;
        istart = n0*i;
        iend = istart +n;
        for(int j = istart; j <iend; j++){
            for(int m = 0; m < k; m++){
                pointer = m*N + j;
                if(gids[pointer] == j) pointer = j;
                fout << sqrt(distances[pointer]) << " ";
            }
            fout<<"\n";
        }
        fout.close();
    }

}

// output labels
void output_labels(const int N, const vector<int> label, const string name)
{

    ofstream fout;
    fout.open(name);
    for(int i = 0; i< N; i++){
        fout<<label[i]<<"\n";
    }
    fout.close();
}



//read labels
vector<int> read_labels(const int N, const string name)
{
    vector<int> label(N);
    ifstream fin(name);
    while(fin.peek() == '%') fin.ignore(2048, '\n');
    if(!fin.is_open()) cout << "not open" << endl;
    int k = 0;
    int j;
    for(int i=0; i<N; i++){
        fin>>j;
        label[k] = j;
        k+=1;
    }
    fin.close();
    return label;

}








