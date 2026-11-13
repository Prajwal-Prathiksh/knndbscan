// Logging for cpp/core and the CLI. Every message goes to one handler: the C API's
// knndbscan_set_log_handler (the Python package forwards to its `logging` loggers), or
// by default a line on standard error. A disabled level costs one comparison at each
// call site, and no call site sits inside a per-point or per-edge loop.
//
// Calls come only from the thread that called knndbscan(), never from inside an OpenMP
// parallel region, so a handler sees messages in order and need not be thread-safe.
#ifndef KNNDBSCAN_LOG_H
#define KNNDBSCAN_LOG_H

#include <cstdarg>
#include <cstdio>

namespace knnlog {

// Values of knndbscan.h's KNNDBSCAN_LOG_* constants.
enum Level { off = 0, info = 1, debug = 2 };
typedef void (*Handler)(int level, const char* message, void* user_data);

inline int level = off;
inline Handler handler = nullptr;
inline void* handler_data = nullptr;
// Set from MPI at the start of knndbscan() (and by the CLI): messages say which rank
// sent them when there is more than one.
inline int rank = 0;
inline int n_ranks = 1;

inline bool enabled(Level wanted) { return level >= wanted; }

#if defined(__GNUC__) || defined(__clang__)
__attribute__((format(printf, 3, 4)))
#endif
inline void write(Level message_level, bool every_rank, const char* format, ...) {
    if (!every_rank && rank != 0) return;
    char text[512];
    int used = 0;
    if (n_ranks > 1)
        used = std::snprintf(text, sizeof text, "rank %d/%d: ", rank, n_ranks);
    va_list arguments;
    va_start(arguments, format);
    std::vsnprintf(text + used, sizeof text - used, format, arguments);
    va_end(arguments);
    if (handler) {
        handler(message_level, text, handler_data);
    } else {
        // One fprintf per line, so lines from different ranks do not interleave
        // mid-line.
        std::fprintf(stderr, "[knndbscan %s] %s\n",
                     message_level == debug ? "debug" : "info", text);
    }
}

}  // namespace knnlog

// Rank 0 only: summaries of collective phases, identical on every rank.
#define KNN_INFO(...)                                        \
    do {                                                     \
        if (knnlog::enabled(knnlog::info))                   \
            knnlog::write(knnlog::info, false, __VA_ARGS__); \
    } while (0)
#define KNN_DEBUG(...)                                        \
    do {                                                      \
        if (knnlog::enabled(knnlog::debug))                   \
            knnlog::write(knnlog::debug, false, __VA_ARGS__); \
    } while (0)
// Every rank: per-rank state, such as the work each rank holds in a distributed phase.
#define KNN_DEBUG_EVERY_RANK(...)                            \
    do {                                                     \
        if (knnlog::enabled(knnlog::debug))                  \
            knnlog::write(knnlog::debug, true, __VA_ARGS__); \
    } while (0)

#endif
