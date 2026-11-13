/* Contract tests of cpp/include/knndbscan.h, written in C so that the header must stay
 * valid C: argument checks, untouched output on failure, per-thread error messages,
 * restored OpenMP settings, concurrent calls and thread-count independence. */
#include <omp.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "knndbscan.h"

static int failures = 0;

#define CHECK(condition)                                                     \
    do {                                                                     \
        if (!(condition)) {                                                  \
            fprintf(stderr, "%s:%d: CHECK(%s) failed\n", __FILE__, __LINE__, \
                    #condition);                                             \
            failures++;                                                      \
        }                                                                    \
    } while (0)

/* Two pairs, each point the other's only neighbor: clusters {0, 1} and {2, 3}. */
static const int32_t pair_neighbors[] = {1, 0, 3, 2};
static const float pair_distances[] = {0.5f, 0.5f, 0.5f, 0.5f};

static int all_equal(const int32_t* values, int n, int32_t expected) {
    for (int i = 0; i < n; i++)
        if (values[i] != expected) return 0;
    return 1;
}

static void test_valid_call_writes_labels_and_clears_the_error(void) {
    int32_t labels[4] = {7, 7, 7, 7};
    int32_t bad_index[] = {9, 0, 3, 2};
    knndbscan_cluster(bad_index, pair_distances, 4, 1, 1.0f, 2, 1, 0, labels);
    CHECK(strlen(knndbscan_last_error()) > 0);
    CHECK(knndbscan_cluster(pair_neighbors, pair_distances, 4, 1, 1.0f, 2, 1, 0,
                            labels) == KNNDBSCAN_OK);
    CHECK(labels[0] == 0 && labels[1] == 0 && labels[2] == 1 && labels[3] == 1);
    CHECK(strcmp(knndbscan_last_error(), "") == 0);
}

static void expect_rejected(const int32_t* neighbors, const float* distances,
                            int32_t n_points, int32_t n_neighbors, float eps,
                            int32_t min_samples, int32_t n_threads,
                            const char* message) {
    int32_t labels[4] = {7, 7, 7, 7};
    int status = knndbscan_cluster(neighbors, distances, n_points, n_neighbors, eps,
                                   min_samples, n_threads, 0, labels);
    CHECK(status == KNNDBSCAN_ERROR_INVALID_ARGUMENT);
    CHECK(all_equal(labels, 4, 7));
    if (strstr(knndbscan_last_error(), message) == NULL) {
        fprintf(stderr, "expected \"%s\" in \"%s\"\n", message, knndbscan_last_error());
        failures++;
    }
}

static void test_rejected_calls_leave_labels_untouched(void) {
    const int32_t self_loop[] = {0, 0, 3, 2};
    const float negative[] = {-0.5f, 0.5f, 0.5f, 0.5f};
    /* Three points with two neighbors each; row 1's distances are out of order. */
    const int32_t triangle[] = {1, 2, 0, 2, 0, 1};
    const float unsorted[] = {0.1f, 0.2f, 0.5f, 0.2f, 0.1f, 0.2f};
    expect_rejected(NULL, pair_distances, 4, 1, 1.0f, 2, 1, "null");
    expect_rejected(pair_neighbors, pair_distances, 0, 1, 1.0f, 2, 1, "n_points");
    expect_rejected(pair_neighbors, pair_distances, 4, 1, 1.0f, 3, 1, "min_samples");
    expect_rejected(pair_neighbors, pair_distances, 4, 1, 1.0f, 1, 1, "min_samples");
    expect_rejected(pair_neighbors, pair_distances, 4, 1, 0.0f, 2, 1, "eps");
    expect_rejected(pair_neighbors, pair_distances, 4, 1, 1.0f, 2, 0, "n_threads");
    expect_rejected(self_loop, pair_distances, 4, 1, 1.0f, 2, 1, "row 0");
    expect_rejected(pair_neighbors, negative, 4, 1, 1.0f, 2, 1, "negative");
    expect_rejected(triangle, unsorted, 3, 2, 1.0f, 2, 1, "row 1: distances are not");
    expect_rejected(pair_neighbors, pair_distances, 65536, 32768, 1.0f, 2, 1, "2^31");
}

static void* fail_in_another_thread(void* message_out) {
    int32_t labels[4];
    /* A thread that has not failed sees no message, whatever other threads did. */
    if (strcmp(knndbscan_last_error(), "") != 0) failures++;
    knndbscan_cluster(pair_neighbors, pair_distances, 4, 1, 1.0f, 2, 0, 0, labels);
    strncpy((char*)message_out, knndbscan_last_error(), 255);
    return NULL;
}

static void test_error_messages_are_per_thread(void) {
    int32_t labels[4];
    char other[256] = "";
    pthread_t thread;
    knndbscan_cluster(pair_neighbors, pair_distances, 4, 1, -1.0f, 2, 1, 0, labels);
    CHECK(strstr(knndbscan_last_error(), "eps") != NULL);
    pthread_create(&thread, NULL, fail_in_another_thread, other);
    pthread_join(thread, NULL);
    CHECK(strstr(other, "n_threads") != NULL);
    CHECK(strstr(knndbscan_last_error(), "eps") != NULL);
}

static void test_openmp_settings_are_restored(void) {
    int32_t labels[4];
    omp_set_num_threads(3);
    omp_set_dynamic(1);
    const int dynamic =
        omp_get_dynamic(); /* 0 if the runtime cannot do dynamic teams */
    CHECK(knndbscan_cluster(pair_neighbors, pair_distances, 4, 1, 1.0f, 2, 2, 0,
                            labels) == KNNDBSCAN_OK);
    CHECK(omp_get_max_threads() == 3);
    CHECK(omp_get_dynamic() == dynamic);
    omp_set_dynamic(0);
}

/* A random directed graph: distinct non-self neighbors, ascending distances. */
enum { N_POINTS = 600, N_NEIGHBORS = 6, MIN_SAMPLES = 6 };
static int32_t graph_neighbors[N_POINTS * N_NEIGHBORS];
static float graph_distances[N_POINTS * N_NEIGHBORS];
static int32_t graph_labels[N_POINTS];

static unsigned next_random(unsigned* state) {
    *state = *state * 1664525u + 1013904223u;
    return *state >> 8;
}

static int compare_floats(const void* a, const void* b) {
    const float x = *(const float*)a, y = *(const float*)b;
    return (x > y) - (x < y);
}

static void make_random_graph(void) {
    unsigned state = 12345u;
    int offsets[N_POINTS - 1];
    for (int i = 0; i < N_POINTS; i++) {
        for (int j = 0; j < N_POINTS - 1; j++) offsets[j] = j + 1;
        for (int j = 0; j < N_NEIGHBORS; j++) { /* partial Fisher-Yates */
            const int pick = j + (int)(next_random(&state) % (N_POINTS - 1 - j));
            const int swap = offsets[j];
            offsets[j] = offsets[pick];
            offsets[pick] = swap;
            graph_neighbors[i * N_NEIGHBORS + j] = (i + offsets[j]) % N_POINTS;
            graph_distances[i * N_NEIGHBORS + j] =
                (next_random(&state) % 10000) / 10000.0f;
        }
        qsort(graph_distances + i * N_NEIGHBORS, N_NEIGHBORS, sizeof(float),
              compare_floats);
    }
}

static int cluster_random_graph(int32_t* labels, int n_threads) {
    return knndbscan_cluster(graph_neighbors, graph_distances, N_POINTS, N_NEIGHBORS,
                             0.75f, MIN_SAMPLES, n_threads, 0, labels);
}

static void test_labels_do_not_depend_on_thread_count(void) {
    int32_t labels[N_POINTS];
    CHECK(cluster_random_graph(graph_labels, 1) == KNNDBSCAN_OK);
    for (int n_threads = 2; n_threads <= 8; n_threads *= 2) {
        CHECK(cluster_random_graph(labels, n_threads) == KNNDBSCAN_OK);
        CHECK(memcmp(labels, graph_labels, sizeof(labels)) == 0);
    }
}

static void* cluster_repeatedly(void* mismatches) {
    int32_t labels[N_POINTS];
    for (int run = 0; run < 10; run++) {
        if (cluster_random_graph(labels, 2) != KNNDBSCAN_OK ||
            memcmp(labels, graph_labels, sizeof(labels)) != 0)
            (*(int*)mismatches)++;
    }
    return NULL;
}

static void test_concurrent_calls_give_the_serial_result(void) {
    enum { N_THREADS = 8 };
    pthread_t threads[N_THREADS];
    int mismatches[N_THREADS] = {0};
    for (int t = 0; t < N_THREADS; t++)
        pthread_create(&threads[t], NULL, cluster_repeatedly, &mismatches[t]);
    for (int t = 0; t < N_THREADS; t++) {
        pthread_join(threads[t], NULL);
        CHECK(mismatches[t] == 0);
    }
}

int main(void) {
    make_random_graph();
    test_valid_call_writes_labels_and_clears_the_error();
    test_rejected_calls_leave_labels_untouched();
    test_error_messages_are_per_thread();
    test_openmp_settings_are_restored();
    test_labels_do_not_depend_on_thread_count();
    test_concurrent_calls_give_the_serial_result();
    if (failures) fprintf(stderr, "%d check(s) failed\n", failures);
    return failures != 0;
}
