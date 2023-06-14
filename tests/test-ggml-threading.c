#include "ggml-threading.h"
#include "ggml.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

// Purposes:
// 1. general overview of the threading behaviors.
// 2. race (dead lock) detection.

// # build
// cd build
//
// # build release:
//   cmake .. && cmake --build . --config Release
//
// # build with sanitize:
//   cmake .. -DLLAMA_SANITIZE_THREAD=ON && cmake --build . --config Release
//
// # run:
// ./bin/test-ggml-threading

// How to turn off the warning on Apple: malloc: nano zone abandoned due to
// inability to reserve vm space?
// ==> export MallocNanoZone=0, no need to rebuild.
// See `nano_init()` from
// https://opensource.apple.com/source/libmalloc/libmalloc-140.40.1/src/nano_malloc.c.auto.html

// How to view the threading debug:
// ==> uncomment `#define GGML_THREADING_DEBUG 1` from file ggml-threading.c

#define UNUSED(x) (void)(x)

#define MAX_N_THREADS 16

static const int n_repeat = 10;

// It's frustrating to use atomic with c11 on Windows, let's replace atomic
// counter with array.
static int work_done_arr[MAX_N_THREADS];

static enum ggml_compute_error
mock_task_runner(struct ggml_compute_params *params, struct ggml_tensor *node) {
    int64_t loops = node->task_profile.dev_flags[1] * 1000 * 1000;
    if (node->task_profile.stages[params->type].parallel) {
        loops /= params->nth;
    }

    volatile int64_t j = 0;
    for (int i = 0; i < loops; i++) {
        j++;
    }

    UNUSED(j);

    work_done_arr[params->ith]++;
    return GGML_COMPUTE_OK;
}

int test_driver(int id, struct ggml_tensor *node, int n_threads) {
    printf("\n[test-ggml-threading] #%d, n_threads: %d\n", id, n_threads);

    for (int i = 0; i < n_threads; i++) {
        work_done_arr[i] = 0;
    }

    bool wait_on_done = (node->task_profile.dev_flags[0] > 0u);

    enum ggml_threading_features features = GGML_THREADING_FEATURE_PERF;
    if (wait_on_done) {
        features |= GGML_THREADING_FEATURE_WAIT_ON_DONE;
    }

    int t0 = (int)ggml_time_us();

    struct ggml_threading_context *ctx =
        ggml_threading_start(n_threads, ggml_threading_graph_compute_thread,
                             mock_task_runner, features, /*stages_time*/ NULL);

    int t1 = (int)ggml_time_us();

    for (int i = 0; i < n_repeat; i++) {
        enum ggml_compute_error err = ggml_threading_compute_tensor(
            ctx, node, /*wdata*/ NULL, /*wsize*/ 0);
        if (err != GGML_COMPUTE_OK) {
            ggml_threading_stop(ctx);
            fprintf(stderr,
                    "ggml_threading_compute_tensor failed with error: %d.\n",
                    err);
            return 1;
        }
    }

    int t2 = (int)ggml_time_us();

    ggml_threading_stop(ctx);

    int t3 = (int)ggml_time_us();

    int expect = 0;
    for (int i = 0; i < 3; i++) {
        struct ggml_task_stage *ts = &node->task_profile.stages[i];
        if (ts->backend != GGML_TASK_BACKEND_NONE) {
            if (ts->parallel) {
                expect += n_threads;
            } else {
                expect++;
            }
        }
    }
    expect *= n_repeat;

    int actual = 0;
    for (int i = 0; i < n_threads; i++) {
        actual += work_done_arr[i];
    }

    uint8_t loops = node->task_profile.dev_flags[1];

    printf("\tloops: %2d million(s), ---wait_on_done---: %d\n\tstage-0: "
           "(parallel: %d, "
           "wait: %d)\n"
           "\tstage-1: (parallel: %d, wait: %d)\n",
           loops, wait_on_done, node->task_profile.stages[0].parallel,
           node->task_profile.stages[0].wait,
           node->task_profile.stages[1].parallel,
           node->task_profile.stages[1].wait);

    if (actual == expect) {
        printf("\tthreading: init %6.3f ms, compute %6.3f ms, cleanup %6.3f "
               "ms, total %6.3f ms\n",
               1.0 * (t1 - t0) / 1000, 1.0 * (t2 - t1) / 1000,
               1.0 * (t3 - t2) / 1000, 1.0 * (t3 - t0) / 1000);
        return 0;
    }

    fprintf(stderr, "\t== failed. expect %d done, actual %d done\n\n", expect,
            actual);

    return 2;
}

static enum ggml_compute_error
mock_task_runner_fallback(struct ggml_compute_params *params,
                          struct ggml_tensor *node) {
    UNUSED(params);
    if (node->backend == GGML_BACKEND_GPU) {
        // ... finally failed to compute in GPU.

        node->backend = GGML_BACKEND_CPU;
        return GGML_COMPUTE_FALLBACK;
    } else {
        return GGML_COMPUTE_OK;
    }
}

// By design, fallback should happen when attempt computing tensor in GPU,
// thus it is not parallelled.
int test_fallback(struct ggml_tensor *node) {
    struct ggml_threading_context *ctx = ggml_threading_start(
        1, ggml_threading_graph_compute_thread, mock_task_runner_fallback,
        /*features*/ GGML_THREADING_FEATURE_NONE, /*stages_time*/ NULL);

    enum ggml_compute_error err =
        ggml_threading_compute_tensor(ctx, node, /*wdata*/ NULL, /*wsize*/ 0);
    if (err == GGML_COMPUTE_FALLBACK) {
        err = ggml_threading_compute_tensor(ctx, node, /*wdata*/ NULL,
                                            /*wsize*/ 0);
    }

    ggml_threading_stop(ctx);
    if (err != GGML_COMPUTE_OK) {
        fprintf(stderr,
                "ggml_threading_compute_tensor failed with error: %d.\n", err);
        return 1;
    }

    return 0;
}

int main(void) {
    ggml_time_init();

    struct ggml_tensor node;
    memset(&node, 0, sizeof(struct ggml_tensor));

    struct ggml_task_stage *stages = node.task_profile.stages;

    stages[0].backend = GGML_TASK_BACKEND_CPU;
    stages[1].backend = GGML_TASK_BACKEND_CPU;
    stages[2].backend = GGML_TASK_BACKEND_NONE;

    int n_passed = 0;
    int n_tests = 0;

    int parallel[3] = {0, 1, 2};

    // In github build actions (windows-latest-cmake and ubuntu-latest-cmake):
    // When n_threads >= 4, the thread init time and compute time suddenly goes
    // down to 100x ~ 1000x slow -- comparing to n_threads == 2.
    //
    // But the tests (n_threads 1, 2, 4, 6) looks sound on my devices:
    // - MacBook air 2013, ubuntu 22.04
    // - MacBook pro 2018, macOS 13.4
    //
    // So I assume the github build host has limited multi-cpu quota.
    // Will skip computing when threading init time is too slow.
    //
    // NOTE: it's observed that when workload is 0 and n_threads >= number of
    // physical cores:
    // - the wait/wakeup time varies much: can be up to tens or hundreds of the
    //   average time, thus greatly punishes those small workloads.
    // - wait_on_done is general faster than wait_now, can be 10x faster.

    int threads_arr[] = {1, 2, 4, 8};
    int threads_arr_len = sizeof(threads_arr) / sizeof(threads_arr[0]);

    // millions of loops.
    uint8_t workload_arr[] = {0u, 1u, 10u};
    int workload_arr_len = sizeof(workload_arr) / sizeof(workload_arr[0]);

    // node.task_profile.dev_flags: byte 0 for wait_on_done, byte 1 for loops.

    for (int x = 0; x < workload_arr_len; x++) {
        node.task_profile.dev_flags[1] = workload_arr[x];

        for (int i = 0; i < threads_arr_len; i++) {
            int n_threads = threads_arr[i];
            if (n_threads > MAX_N_THREADS) {
                abort();
            }

            printf("\n[test-ggml-threading] ==== n_nodes: %d, n_threads: %d, "
                   "loops: %2d million(s) ====\n",
                   n_repeat, n_threads, workload_arr[x]);

            if (n_threads > 1) { // skip this n_threads when too slow.
                int t0 = (int)ggml_time_us();

                struct ggml_threading_context *ctx = ggml_threading_start(
                    n_threads, ggml_threading_graph_compute_thread,
                    mock_task_runner, 0, /*stages_time*/ NULL);

                int t1 = (int)ggml_time_us();

                ggml_threading_stop(ctx);

                int elapsed_us = t1 - t0;
                if (elapsed_us > 500 * n_threads) {
                    fprintf(stderr,
                            "[test-ggml-threading] warning: it took took %.3f "
                            "ms to start %d worker thread(s).\n",
                            1.0 * elapsed_us / 1000, n_threads - 1);
                    fprintf(stderr, "[test-ggml-threading] warning: looks like "
                                    "the environment is too slow to run this "
                                    "number of threads, skip.\n");
                    continue;
                }
            }

            // multi-threads: parallel + wait_now/wait_on_done

            if (n_threads == 1) {
                stages[0].parallel = false;
                stages[1].parallel = false;
                stages[0].wait = false;
                stages[1].wait = false;

                n_tests++;
                if (test_driver(n_tests, &node, n_threads) == 0) {
                    n_passed++;
                }
                continue;
            }

            for (int j = 0; j < 3; j++) {
                stages[0].wait = false;
                stages[1].wait = false;
                node.task_profile.dev_flags[0] = 0u;

                if (parallel[j] == 0) {
                    stages[0].parallel = false;
                    stages[1].parallel = false;

                    n_tests++;
                    if (test_driver(n_tests, &node, n_threads) == 0) {
                        n_passed++;
                    }
                } else if (parallel[j] == 1) {
                    stages[0].parallel = true;
                    stages[1].parallel = false;

                    for (int k = 0; k < 2; k++) {
                        stages[1].wait = (k == 1);

                        if (!stages[1].wait) {
                            n_tests++;
                            if (test_driver(n_tests, &node, n_threads) == 0) {
                                n_passed++;
                            }
                            continue;
                        }

                        // wait

                        for (int m = 0; m < 2; m++) {
                            if (m == 1) {
                                node.task_profile.dev_flags[0] = 1u;
                            }
                            n_tests++;
                            if (test_driver(n_tests, &node, n_threads) == 0) {
                                n_passed++;
                            }
                            node.task_profile.dev_flags[0] = 0u;
                        }
                    }
                } else {
                    stages[0].parallel = true;
                    stages[1].parallel = true;

                    n_tests++;
                    if (test_driver(n_tests, &node, n_threads) == 0) {
                        n_passed++;
                    }
                }
            }
        }
    }

    {
        ++n_tests;

        node.backend = GGML_BACKEND_GPU;
        if (test_fallback(&node) == 0) {
            ++n_passed;
            printf("\n[test-ggml-threading] test fallback: ok\n\n");
        }
    }

    printf("[test-ggml-threading] %d/%d passed.\n", n_passed, n_tests);

    return (n_passed == n_tests) ? 0 : 1;
}
