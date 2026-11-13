// Implementation of knndbscan.h: checks the graph against the core's assumptions, then
// runs cpp/core under a lock. The only caller of knndbscan() besides the CLI.
#include <omp.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <exception>
#include <limits>
#include <mutex>
#include <string>
#include <vector>

#include "clusters.h"
#include "knndbscan.h"
#include "log.h"

static_assert(sizeof(point_int) == sizeof(int32_t), "the C API passes labels as int32");
static_assert(KNNDBSCAN_LOG_OFF == knnlog::off && KNNDBSCAN_LOG_INFO == knnlog::info &&
                  KNNDBSCAN_LOG_DEBUG == knnlog::debug,
              "the C API passes knnlog levels");

namespace {

// The core keeps its state in globals, so concurrent calls would corrupt each other.
std::mutex call_mutex;
thread_local std::string last_error;

int fail(int code, std::string message) {
    last_error = std::move(message);
    return code;
}

// Why row i breaks an assumption of the core, or nullptr if it is valid. Unsorted rows
// and self-loops matter as much as bad indices: the border-labelling loop stops at the
// first distance >= eps and relies on sorted rows to stay inside its row.
const char* row_problem(int64_t i, int32_t n_points, int32_t n_neighbors,
                        const int32_t* neighbors, const float* distances) {
    const int32_t* row = neighbors + i * n_neighbors;
    const float* dist = distances + i * n_neighbors;
    for (int32_t j = 0; j < n_neighbors; ++j) {
        if (row[j] < 0 || row[j] >= n_points)
            return "neighbor index outside [0, n_points)";
        if (row[j] == i) return "the point is listed as its own neighbor";
        if (!std::isfinite(dist[j]) || dist[j] < 0.0f)
            return "distance is negative, NaN or infinite";
        if (j > 0 && dist[j] < dist[j - 1])
            return "distances are not in ascending order";
    }
    return nullptr;
}

int check_arguments(const int32_t* neighbors, const float* distances, int32_t n_points,
                    int32_t n_neighbors, float eps, int32_t min_samples,
                    int32_t n_threads, int32_t log_level, const int32_t* labels) {
    const int code = KNNDBSCAN_ERROR_INVALID_ARGUMENT;
    if (!neighbors || !distances || !labels) return fail(code, "null array pointer");
    if (n_points < 1) return fail(code, "n_points must be at least 1");
    if (n_neighbors < 1) return fail(code, "n_neighbors must be at least 1");
    if (static_cast<int64_t>(n_points) * n_neighbors >
        std::numeric_limits<int32_t>::max())
        return fail(code, "n_points * n_neighbors must be below 2^31");
    if (min_samples < 2 || min_samples > n_neighbors + 1)
        return fail(code, "min_samples must be in [2, n_neighbors + 1], got " +
                              std::to_string(min_samples) +
                              " with n_neighbors = " + std::to_string(n_neighbors));
    if (!std::isfinite(eps) || eps <= 0.0f)
        return fail(code, "eps must be positive and finite");
    if (n_threads < 1) return fail(code, "n_threads must be at least 1");
    if (log_level < KNNDBSCAN_LOG_OFF || log_level > KNNDBSCAN_LOG_DEBUG)
        return fail(code, "log_level must be a KNNDBSCAN_LOG_* value");

    int64_t first_bad = n_points;
#pragma omp parallel for num_threads(n_threads) reduction(min : first_bad)
    for (int64_t i = 0; i < n_points; ++i) {
        if (row_problem(i, n_points, n_neighbors, neighbors, distances))
            first_bad = std::min(first_bad, i);
    }
    if (first_bad < n_points)
        return fail(code, "row " + std::to_string(first_bad) + ": " +
                              row_problem(first_bad, n_points, n_neighbors, neighbors,
                                          distances));
    return KNNDBSCAN_OK;
}

// The core's cluster numbers follow root orders that depend on thread scheduling, so
// the same partition can come out numbered differently. Numbering clusters by their
// lowest-indexed point makes labels a function of the partition alone.
void renumber_by_first_point(const std::vector<point_int>& core_labels,
                             int32_t* labels) {
    const point_int max_label =
        *std::max_element(core_labels.begin(), core_labels.end());
    std::vector<int32_t> renumbered(static_cast<size_t>(max_label) + 1, -1);
    int32_t next = 0;
    for (size_t i = 0; i < core_labels.size(); ++i) {
        const point_int label = core_labels[i];
        if (label < 0) {
            labels[i] = -1;
            continue;
        }
        if (renumbered[label] < 0) renumbered[label] = next++;
        labels[i] = renumbered[label];
    }
}

}  // namespace

extern "C" int knndbscan_cluster(const int32_t* neighbors, const float* distances,
                                 int32_t n_points, int32_t n_neighbors, float eps,
                                 int32_t min_samples, int32_t n_threads,
                                 int32_t log_level, int32_t* labels) {
    last_error.clear();
    int status = check_arguments(neighbors, distances, n_points, n_neighbors, eps,
                                 min_samples, n_threads, log_level, labels);
    if (status != KNNDBSCAN_OK) return status;

    std::lock_guard<std::mutex> lock(call_mutex);
    // The core sizes per-thread buffers from omp_get_num_threads(), so dynamic teams
    // must be off; both settings are per calling thread and restored afterwards.
    const int previous_threads = omp_get_max_threads();
    const int previous_dynamic = omp_get_dynamic();
    omp_set_dynamic(0);
    omp_set_num_threads(n_threads);
    knnlog::level = log_level;
    try {
        std::vector<point_int> result =
            knndbscan(n_points, eps, min_samples, n_neighbors, neighbors, distances);
        renumber_by_first_point(result, labels);
    } catch (const std::exception& error) {
        status = fail(KNNDBSCAN_ERROR_INTERNAL, error.what());
    } catch (...) {
        status = fail(KNNDBSCAN_ERROR_INTERNAL, "unknown error in the clustering core");
    }
    knnlog::level = knnlog::off;
    omp_set_num_threads(previous_threads);
    omp_set_dynamic(previous_dynamic);
    return status;
}

extern "C" void knndbscan_set_log_handler(knndbscan_log_handler handler,
                                          void* user_data) {
    std::lock_guard<std::mutex> lock(call_mutex);
    knnlog::handler = handler;
    knnlog::handler_data = user_data;
}

extern "C" const char* knndbscan_last_error(void) { return last_error.c_str(); }
