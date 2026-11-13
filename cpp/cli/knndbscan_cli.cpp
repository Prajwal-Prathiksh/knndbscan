// Command-line kNN-DBSCAN on MPI ranks: each rank reads its block of rows from a
// kNN-graph text file, and rank 0 writes one label per line.
#include <getopt.h>

#include <cstring>

#include "core.h"
#include "log.h"

using namespace std;

void prepare_read(const int gsize, const int rank, const point_int N, point_int& n,
                  point_int& rstart, point_int& rend,
                  vector<point_int>& num_points_rank) {
    point_int n0, n1, p;
    n0 = ceil((double)N / (double)gsize);
    p = ceil((double)N / (double)n0);
    n1 = N - n0 * (p - 1);
    n = n0;
    rstart = n0 * rank;
    if (rank == p - 1) {
        n = n1;
    } else if (rank > p - 1) {
        n = 0;
        rstart = N;
    }
    rend = rstart + n;
    fill(num_points_rank.begin(), num_points_rank.end(), 0);
    for (int i = 0; i < p; i++) {
        num_points_rank[i] = (i < p - 1) ? n0 : n1;
    }
}

// "off", "info", "debug" or 0-2; -1 for anything else.
static int parse_log_level(const char* text) {
    if (!strcmp(text, "off") || !strcmp(text, "0")) return knnlog::off;
    if (!strcmp(text, "info") || !strcmp(text, "1")) return knnlog::info;
    if (!strcmp(text, "debug") || !strcmp(text, "2")) return knnlog::debug;
    return -1;
}

static void show_usage(char* argv0) {
    const char* params =
        "Usage: %s [switches] -n num_points -e epsilon -m minPts -i knngraph_name -k "
        "num_neighbors -o output_name -t threads\n"
        "   -n num_points  : number of points\n"
        "   -e epsilon     : input parameter of kNN-DBSCAN, raidus of consideration "
        "for core points\n"
        "   -m minpts      : input parameter of kNN-DBSCAN, min points\n"
        "   -i knngraph_name    : file containing kNN graph\n"
        "   -k num_neighbors    : number of neighbors for each point in input file,k "
        "should be no less than m\n"
        "   -o output      : output file with label for each point, the order of "
        "points is as the same with knn graph\n"
        "   -t threads  : number of threads per MPI task\n"
        "   -l level    : log level on standard error: off, info or debug (default:\n"
        "                 $KNNDBSCAN_LOG_LEVEL, else info)\n\n";

    fprintf(stderr, params, argv0);
    exit(-1);
}

int main(int argc, char** argv) {
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
    const char* log_setting = getenv("KNNDBSCAN_LOG_LEVEL");
    int log_level = log_setting ? parse_log_level(log_setting) : knnlog::info;

    while ((opt = getopt(argc, argv, "n:e:m:i:k:o:t:l:")) != EOF) {
        switch (opt) {
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
            case 'l':
                log_level = parse_log_level(optarg);
                break;
            case '?':
                show_usage(argv[0]);
                break;
            default:
                show_usage(argv[0]);
                break;
        }
    }
    if (N < 0 || eps < 1.e-6 || minPts < 0 || nthreads < 1 || knn_file == NULL ||
        log_level < 0) {
        show_usage(argv[0]);
        exit(-1);
    }

    if (graph_k < 0) graph_k = minPts;
    omp_set_dynamic(0);  // Explicitly disable dynamic teams
    omp_set_num_threads(nthreads);
    int k = minPts - 1;  // number stored for each point, we neglect 1st neighbor
#pragma omp parallel
    {
        nthreads = omp_get_num_threads();
    }

    knnlog::level = log_level;
    knnlog::rank = rank;
    knnlog::n_ranks = gsize;
    KNN_INFO("reading %s: %d points, %d neighbor column(s) per line", knn_file, N,
             graph_k);
    const double t_start = MPI_Wtime();

    // ---------- read kNN graph ------------------
    point_int n, rstart, rend;
    vector<point_int> num_points_rank(gsize);
    prepare_read(gsize, rank, N, n, rstart, rend, num_points_rank);
    edge_int nE = n * k;
    point_int* JA = (point_int*)calloc(nE, sizeof(point_int));
    float* A = (float*)calloc(nE, sizeof(float));

    ifstream fin(knn_file);
    string line;
    point_int gid, id, id_1st;
    edge_int count = 0;
    float dist, dist_1st;
    for (point_int i = 0; i < rstart; i++) getline(fin, line);
    for (point_int i = 0; i < n; i++) {
        fin >> gid;
        fin >> dist_1st >> id_1st;
        for (int j = 0; j < graph_k - 1; j++) {
            fin >> dist >> id;
            if (j < k) {
                // With duplicate points the first column may be a duplicate rather than
                // the point itself; the point then appears later and takes its place.
                JA[count] = (id == gid) ? id_1st : id;
                A[count] = dist;
                count += 1;
            }
        }
    }

    fin.close();
    MPI_Barrier(MPI_COMM_WORLD);
    KNN_INFO("read the kNN graph in %.3e s", MPI_Wtime() - t_start);

    // ---------- knn-dbscan clustering ------------------
    vector<point_int> R;
    R = knndbscan(N, eps, minPts, k, JA, A);

    // ---------- output labels ------------------
    int* rcounts = new int[gsize];
    int* rdispls = new int[gsize];
    rdispls[0] = 0;
    for (int i = 0; i < gsize; i++) {
        rcounts[i] = num_points_rank[i];
        if (i != 0) rdispls[i] = rdispls[i - 1] + rcounts[i - 1];
    }
    vector<int> labels;
    if (rank == 0) labels.resize(N);
    MPI_Gatherv(R.data(), n, MPI_INT, labels.data(), rcounts, rdispls, MPI_INT, 0,
                MPI_COMM_WORLD);
    // Only rank 0 may open the file: opening truncates it, so a rank that opened it
    // after rank 0 finished writing would leave it empty.
    if (rank == 0) {
        ofstream fout(label_file);
        for (int i = 0; i < N; i++) {
            fout << labels[i] << "\n";
        }
        fout.close();
    }

    KNN_INFO("wrote %s; %.3e s in total", label_file, MPI_Wtime() - t_start);
    MPI_Finalize();
    return 0;
}
