/* Public C API of the knndbscan shared library: kNN-DBSCAN clustering of a precomputed
 * kNN graph. The Python package calls it through ctypes (src/knndbscan/_native.py). */
#ifndef KNNDBSCAN_H
#define KNNDBSCAN_H

#include <stdint.h>

#if defined(__GNUC__) || defined(__clang__)
#define KNNDBSCAN_API __attribute__((visibility("default")))
#else
#define KNNDBSCAN_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

enum {
    KNNDBSCAN_OK = 0,
    /* An argument or the graph violates the contract below; nothing was computed. */
    KNNDBSCAN_ERROR_INVALID_ARGUMENT = 1,
    /* The algorithm failed (for example, out of memory). */
    KNNDBSCAN_ERROR_INTERNAL = 2,
};

/* Log levels, from least to most verbose. INFO reports each phase of a call (core
 * points, subtrees, clusters, timings); DEBUG adds every Boruvka step and per-rank
 * detail. */
enum {
    KNNDBSCAN_LOG_OFF = 0,
    KNNDBSCAN_LOG_INFO = 1,
    KNNDBSCAN_LOG_DEBUG = 2,
};

/* Receives one log message (without a trailing newline) and the `user_data` given to
 * knndbscan_set_log_handler. Called only from the thread inside knndbscan_cluster, and
 * the message is valid only during the call. */
typedef void (*knndbscan_log_handler)(int level, const char* message, void* user_data);

/* Send log messages to `handler`, or with NULL to standard error (the default). Takes
 * effect for later knndbscan_cluster calls; waits for a running call to finish. */
KNNDBSCAN_API void knndbscan_set_log_handler(knndbscan_log_handler handler,
                                             void* user_data);

/* Cluster a kNN graph and write one label per point into `labels`.
 *
 * Graph: `neighbors` and `distances` are row-major (n_points x n_neighbors) arrays. Row
 * i lists the neighbors of point i in ascending order of distance and does not contain
 * i. Indices must lie in [0, n_points); distances must be finite and non-negative. All
 * of this is checked before clustering, because the algorithm would otherwise read out
 * of bounds or loop forever.
 *
 * Parameters: a point is a core point when the distance to its (min_samples - 1)-th
 * neighbor is at most eps; 2 <= min_samples <= n_neighbors + 1. Core points are linked
 * through their first min_samples - 1 neighbors; later columns are not read. A border
 * point joins the cluster of its nearest core neighbor within eps. n_threads >= 1
 * OpenMP threads are used; the caller's OpenMP settings are restored on return.
 * `log_level` (a KNNDBSCAN_LOG_* value) selects the messages sent to the log handler;
 * with KNNDBSCAN_LOG_OFF nothing is formatted or written.
 *
 * Limits: n_points * n_neighbors must be below 2^31.
 *
 * Output: `labels` (n_points entries, caller-owned) receives 0..n_clusters-1 for points
 * in clusters, numbered in order of each cluster's lowest-indexed point, and -1 for
 * noise; the result therefore does not depend on n_threads. It is written only when
 * the call returns KNNDBSCAN_OK.
 *
 * Thread safety: calls from several threads are serialized internally.
 *
 * Returns KNNDBSCAN_OK or an error code; knndbscan_last_error() then describes the
 * failure. */
KNNDBSCAN_API int knndbscan_cluster(const int32_t* neighbors, const float* distances,
                                    int32_t n_points, int32_t n_neighbors, float eps,
                                    int32_t min_samples, int32_t n_threads,
                                    int32_t log_level, int32_t* labels);

/* Message for the calling thread's most recent failed call, or "" if none has failed.
 * The pointer stays valid until the thread's next call into this library. */
KNNDBSCAN_API const char* knndbscan_last_error(void);

#ifdef __cplusplus
}
#endif

#endif
