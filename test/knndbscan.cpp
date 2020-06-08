#include "../include/globals.h"
#include "../src/clusters.cpp"

using namespace std;

void prepare_read(const int gsize, const int rank, const point_int N, point_int &n, point_int &rstart, point_int &rend, vector<point_int> &num_points_rank)
{
    point_int n0, n1, p;
    n0 = ceil((double)N / (double)gsize);
    p = ceil((double)N / (double)n0);
    n1 = N - n0*(p-1);
    n = n0;
    rstart = n0 * rank;
    if(rank == p - 1){
       n = n1;
    }else if(rank > p - 1){
       n = 0;
       rstart = N;
    }
    rend = rstart + n;
    fill(num_points_rank.begin(), num_points_rank.end(), 0);
    for (int i = 0; i< p; i++){
        num_points_rank[i] = (i < p-1) ? n0 : n1;
    }
}

static void show_usage(char *argv0)
{
    const char *params =
    "Usage: %s [switches] -n num_points -e epsilon -m minPts -i knngraph_name -k num_neighbors -o output_name -t threads\n"
    "   -n num_points  : number of points\n"
    "   -e epsilon     : input parameter of kNN-DBSCAN, raidus of consideration for core points\n"
    "   -m minpts      : input parameter of kNN-DBSCAN, min points\n"
    "   -i knngraph_name    : file containing kNN graph\n"
    "   -k num_neighbors    : number of neighbors for each point in input file,k should be no less than m\n"
    "   -o output      : output file with label for each point, the order of points is as the same with knn graph\n"
    "   -t threads  : number of threads per MPI task\n\n";

    fprintf(stderr, params, argv0);
    exit(-1);
}


int main(int argc, char **argv) 
{

    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int rank, gsize;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &gsize);

    // initialize
    int opt;
    point_int N = -1; 
    float eps = -1.0; 
    int minPts = -1; 
    int graph_k = -1;
    int nthreads = -1; 
    char* knn_file = NULL;
    char* label_file = NULL;

    while ((opt=getopt(argc,argv,"n:e:m:i:k:o:t:d:p:v:z:bxghncul"))!= EOF)
    {
        switch (opt)
        {
            case 'n':
                N = atoi(optarg);
                break;
            case 'e':
                eps = atof(optarg);
                break;
            case 'm':
                minPts = atoi(optarg);  
                break;
            case 'i':
                knn_file = optarg;
                break;
            case 'k':
                graph_k = atoi(optarg);
                break;
            case 'o':
                label_file = optarg;
                break;
            case 't':
                nthreads = atoi(optarg);
                break;
            case '?':
                show_usage(argv[0]);
                break;
            default:
                show_usage(argv[0]);
                break;
        }
    }
    if(N < 0 || eps < 1.e-6 || minPts<0 || nthreads<1 || knn_file ==NULL)
    {
        show_usage(argv[0]);    
        exit(-1);
    }

    if(graph_k < 0) graph_k = minPts; 
    omp_set_dynamic(0);     // Explicitly disable dynamic teams
    omp_set_num_threads(nthreads);
    int k = minPts-1; //number stored for each point, we neglect 1st neighbor
    #pragma omp parallel
    {
        nthreads = omp_get_num_threads();
    }
    if(rank == 0){
        cout<< "number of process: "<< gsize<<endl;
        cout<< "threads per MPI task: "<< nthreads<<endl;
        cout<< "total number of points: "<< N<< endl;
        cout<< "number of neighbors in knn graph: " << graph_k <<endl;
        cout<< "input clustering parameters: eps="<<eps<<" ,minPts="<<minPts<<endl;
    }

    // ---------- read kNN graph ------------------
    point_int n, rstart, rend;    
    vector<point_int> num_points_rank(gsize);
    prepare_read(gsize, rank, N, n, rstart, rend, num_points_rank);
    edge_int nE = n * k;
    point_int *JA = (point_int*)calloc(nE, sizeof(point_int));
    float *A = (float*)calloc(nE, sizeof(float));
    
    ifstream fin(knn_file);
    string line;
    point_int gid, id, id_1st;
    edge_int count = 0;
    float dist, dist_1st;
    for(point_int i=0; i<rstart; i++) getline(fin, line);
    for(point_int i=0; i<n; i++){
        fin >> gid;
        fin >> dist_1st >> id_1st;
        for(int j = 0; j < graph_k - 1; j++){
            fin >> dist >> id;
            if( j < k){
                JA[count] = id;
                A[count] = dist;
                count +=1;
            }
            if(id == gid)  JA[count] = id_1st; 
        }
    }   

    fin.close();
    MPI_Barrier(MPI_COMM_WORLD);
    if(rank == 0) cout<<"finish kNN-G read."<<endl;

    // ---------- knn-dbscan clustering ------------------
    vector<point_int> R;
    R = knndbscan(N, eps, minPts, k, JA, A);
    if(rank == 0) cout<<"finish knn dbscan."<<endl;

    // ---------- output labels ------------------
    int* rcounts = new int[gsize];
    int* rdispls = new int[gsize];
    rdispls[0] = 0;
    for (int i = 0; i < gsize; i++){
        rcounts[i] = num_points_rank[i];
        if (i != 0) rdispls[i] = rdispls[i-1] + rcounts[i-1];
    }
    vector<int> labels;
    if(rank == 0) labels.resize(N);
    MPI_Gatherv(&R[0], n, MPI_INT, &labels[0], rcounts, rdispls, MPI_INT, 0,  MPI_COMM_WORLD);
    ofstream fout;
    fout.open(label_file);
    if(rank ==0){
        for(int i = 0; i< N; i++){
            fout<<labels[i]<<"\n";
        }
        fout.close();
    }

    if(rank == 0) cout<<"finish label write."<<endl;
    MPI_Finalize();
    return 0;
 
}





