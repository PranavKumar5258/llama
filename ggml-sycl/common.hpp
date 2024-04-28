//
// MIT license
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: MIT
//

//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

#ifndef GGML_SYCL_COMMON_HPP
#define GGML_SYCL_COMMON_HPP

#include <fstream>
#include <iostream>

#include "dpct/helper.hpp"

#define GGML_COMMON_DECL_SYCL
#define GGML_COMMON_IMPL_SYCL
#include "ggml-common.h"

void* ggml_sycl_host_malloc(size_t size);
void ggml_sycl_host_free(void* ptr);

static int g_ggml_sycl_debug = 0;
#define GGML_SYCL_DEBUG(...)        \
  do {                              \
    if (g_ggml_sycl_debug)          \
      fprintf(stderr, __VA_ARGS__); \
  } while (0)

#define CHECK_TRY_ERROR(expr)                                            \
  [&]() {                                                                \
    try {                                                                \
      expr;                                                              \
      return dpct::success;                                              \
    } catch (std::exception const& e) {                                  \
      std::cerr << e.what() << "\nException caught at file:" << __FILE__ \
                << ", line:" << __LINE__ << ", func:" << __func__        \
                << std::endl;                                            \
      return dpct::default_error;                                        \
    }                                                                    \
  }()

// #define DEBUG_SYCL_MALLOC

static int g_work_group_size = 0;
// typedef sycl::half ggml_fp16_t;

#define __SYCL_ARCH__ DPCT_COMPATIBILITY_TEMP
#define VER_4VEC 610 // todo for hardward optimize.
#define VER_GEN9 700 // todo for hardward optimize.
#define VER_GEN12 1000000 // todo for hardward optimize.
#define VER_GEN13 (VER_GEN12 + 1030) // todo for hardward optimize.

#define GGML_SYCL_MAX_NODES 8192 // TODO: adapt to hardwares

// define for XMX in Intel GPU
// TODO: currently, it's not used for XMX really.
#define SYCL_USE_XMX

// max batch size to use MMQ kernels when tensor cores are available
#define XMX_MAX_BATCH_SIZE 32

#if defined(_MSC_VER)
#pragma warning(disable : 4244 4267) // possible loss of data
#endif

// dmmv = dequantize_mul_mat_vec
#ifndef GGML_SYCL_DMMV_X
#define GGML_SYCL_DMMV_X 32
#endif
#ifndef GGML_SYCL_MMV_Y
#define GGML_SYCL_MMV_Y 1
#endif

enum ggml_sycl_backend_gpu_mode {
  SYCL_UNSET_GPU_MODE = -1,
  SYCL_SINGLE_GPU_MODE = 0,
  SYCL_MUL_GPU_MODE
};

static_assert(sizeof(sycl::half) == sizeof(ggml_fp16_t), "wrong fp16 size");

static void crash() {
  int* ptr = NULL;
  *ptr = 0;
}

static void ggml_sycl_error(
    const char* stmt,
    const char* func,
    const char* file,
    const int line,
    const char* msg) {
  fprintf(stderr, "SYCL error: %s: %s\n", stmt, msg);
  fprintf(stderr, "  in function %s at %s:%d\n", func, file, line);
  GGML_ASSERT(!"SYCL error");
}

#define SYCL_CHECK(err)                     \
  do {                                      \
    auto err_ = (err);                      \
    if (err_ != 0)                          \
      ggml_sycl_error(                      \
          #err,                             \
          __func__,                         \
          __FILE__,                         \
          __LINE__,                         \
          "Meet error in this line code!"); \
  } while (0)

#if DPCT_COMPAT_RT_VERSION >= 11100
#define GGML_SYCL_ASSUME(x) __builtin_assume(x)
#else
#define GGML_SYCL_ASSUME(x)
#endif // DPCT_COMPAT_RT_VERSION >= 11100

#ifdef GGML_SYCL_F16
typedef sycl::half dfloat; // dequantize float
typedef sycl::half2 dfloat2;
#else
typedef float dfloat; // dequantize float
typedef sycl::float2 dfloat2;
#endif // GGML_SYCL_F16

#define WARP_SIZE 32
#define MATRIX_ROW_PADDING \
  512 // last row of quant. matrices is a multiple of this to avoid
      // out-of-bounds memory accesses

#define MMVQ_MAX_BATCH_SIZE  8

static const int8_t kvalues_iq4nl[16]={-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};

#define SYCL_GELU_BLOCK_SIZE 256
#define SYCL_SILU_BLOCK_SIZE 256
#define SYCL_TANH_BLOCK_SIZE 256
#define SYCL_RELU_BLOCK_SIZE 256
#define SYCL_HARDSIGMOID_BLOCK_SIZE 256
#define SYCL_HARDSWISH_BLOCK_SIZE 256
#define SYCL_SQR_BLOCK_SIZE 256
#define SYCL_CPY_BLOCK_SIZE 32
#define SYCL_SCALE_BLOCK_SIZE 256
#define SYCL_CLAMP_BLOCK_SIZE 256
#define SYCL_ROPE_BLOCK_SIZE 256
#define SYCL_SOFT_MAX_BLOCK_SIZE 1024
#define SYCL_ALIBI_BLOCK_SIZE 32
#define SYCL_DIAG_MASK_INF_BLOCK_SIZE 32
#define SYCL_QUANTIZE_BLOCK_SIZE 256
#define SYCL_DEQUANTIZE_BLOCK_SIZE 256
#define SYCL_GET_ROWS_BLOCK_SIZE 256
#define SYCL_UPSCALE_BLOCK_SIZE 256
#define SYCL_CONCAT_BLOCK_SIZE 256
#define SYCL_PAD_BLOCK_SIZE 256
#define SYCL_ACC_BLOCK_SIZE 256
#define SYCL_IM2COL_BLOCK_SIZE 256
#define SYCL_POOL2D_BLOCK_SIZE 256

// dmmv = dequantize_mul_mat_vec
#ifndef GGML_SYCL_DMMV_X
#define GGML_SYCL_DMMV_X 32
#endif
#ifndef GGML_SYCL_MMV_Y
#define GGML_SYCL_MMV_Y 1
#endif

#ifndef K_QUANTS_PER_ITERATION
#define K_QUANTS_PER_ITERATION 2
#else
static_assert(
    K_QUANTS_PER_ITERATION == 1 || K_QUANTS_PER_ITERATION == 2,
    "K_QUANTS_PER_ITERATION must be 1 or 2");
#endif

#ifndef GGML_SYCL_PEER_MAX_BATCH_SIZE
#define GGML_SYCL_PEER_MAX_BATCH_SIZE 128
#endif // GGML_SYCL_PEER_MAX_BATCH_SIZE

#define MUL_MAT_SRC1_COL_STRIDE 128
#define MAX_STREAMS 8
#define SYCL_MAX_DEVICES 48

static dpct::queue_ptr g_syclStreams[SYCL_MAX_DEVICES][MAX_STREAMS] = {{0}};

struct ggml_tensor_extra_gpu {
  void* data_device[SYCL_MAX_DEVICES]; // 1 pointer for each device for split
                                       // tensors
  dpct::event_ptr events[SYCL_MAX_DEVICES]
                        [MAX_STREAMS]; // events for synchronizing multiple GPUs
};

class sycl_gpu_mgr {
 public:
  std::vector<int> gpus;
  std::vector<sycl::device> devices;
  sycl::queue* first_queue;
  sycl::context co_ctx;
  int max_compute_units = 0;
  int work_group_size = 0;
  std::string gpus_list = "";

  /*
  Use all GPUs with same top max compute units
  */
  sycl_gpu_mgr() {
    detect_sycl_gpu_list_with_max_cu();
    get_allow_gpus();
    create_context_with_gpus();
  }

  /*
  Only use the assigned GPU
  */
  sycl_gpu_mgr(int main_gpu_id) {
    sycl::device device = dpct::dev_mgr::instance().get_device(main_gpu_id);
    dpct::device_info prop;
    dpct::get_device_info(prop, device);
    gpus.push_back(main_gpu_id);
    devices.push_back(device);
    work_group_size = prop.get_max_work_group_size();
    max_compute_units = prop.get_max_compute_units();

    get_allow_gpus();
    create_context_with_gpus();
  }

  void create_context_with_gpus() {
    sycl::context ctx = sycl::context(devices);
    assert(gpus.size() > 0);
    first_queue = dpct::get_current_device().create_queue(ctx, devices[0]);
    co_ctx = first_queue->get_context();
  }

  sycl::context& get_co_ctx() {
    return co_ctx;
  }

  void get_allow_gpus() {
    gpus_list = "";
    for (size_t i = 0; i < gpus.size(); ++i) {
      gpus_list += std::to_string(gpus[i]);
      gpus_list += ",";
    }
    if (gpus_list.length() > 1) {
      gpus_list.pop_back();
    }
  }

  bool is_allowed_gpu(int device_id) {
    return std::find(gpus.begin(), gpus.end(), device_id) != gpus.end();
  }

  void detect_sycl_gpu_list_with_max_cu() try {
    int device_count = dpct::dev_mgr::instance().device_count();

    for (int id = 0; id < device_count; id++) {
      sycl::device device = dpct::dev_mgr::instance().get_device(id);
      if (!device.is_gpu())
        continue;
      dpct::device_info prop;
      dpct::get_device_info(prop, device);
      if (max_compute_units < prop.get_max_compute_units())
        max_compute_units = prop.get_max_compute_units();
    }

    for (int id = 0; id < device_count; id++) {
      sycl::device device = dpct::dev_mgr::instance().get_device(id);
      if (!device.is_gpu())
        continue;
      dpct::device_info prop;
      dpct::get_device_info(prop, device);
      if (max_compute_units == prop.get_max_compute_units() &&
          is_ext_oneapi_device(device)) {
        gpus.push_back(id);
        devices.push_back(device);
        work_group_size = prop.get_max_work_group_size();
      }
    }
    return;
  } catch (sycl::exception const& exc) {
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__
              << ", line:" << __LINE__ << std::endl;
    std::exit(1);
  }

  int get_gpu_count() {
    return (int)gpus.size();
  }

  int get_index(int id) {
    for (int i = 0; i < (int)gpus.size(); i++) {
      if (gpus[i] == id)
        return i;
    }
    printf("miss to get device index by id=%d\n", id);
    GGML_ASSERT(false);
  }

  int get_next_index(int id) {
    int cur_index = get_index(id);
    for (int i = cur_index + 1; i < (int)gpus.size(); i++) {
      if (gpus[i] == id)
        return i;
    }
    GGML_ASSERT(false);
  }

  bool is_ext_oneapi_device(const sycl::device& dev) {
    sycl::backend dev_backend = dev.get_backend();
    if (dev_backend == sycl::backend::ext_oneapi_level_zero ||
        dev_backend == sycl::backend::ext_oneapi_cuda ||
        dev_backend == sycl::backend::ext_oneapi_hip)
      return true;
    return false;
  }
};

static sycl_gpu_mgr* g_sycl_gpu_mgr = NULL;
static int g_device_count = -1;
static int g_all_sycl_device_count = -1;
static int g_main_device = -1;
static int g_main_device_id = -1;
static bool g_ggml_backend_sycl_buffer_type_initialized = false;

static std::array<float, SYCL_MAX_DEVICES> g_default_tensor_split = {};

static float g_tensor_split[SYCL_MAX_DEVICES] = {0};

static ggml_sycl_backend_gpu_mode g_ggml_sycl_backend_gpu_mode =
    SYCL_UNSET_GPU_MODE;

struct sycl_device_capabilities {
  int cc; // compute capability
  bool vmm; // virtual memory support
  size_t vmm_granularity; // granularity of virtual memory
  int device_id;
};

static sycl_device_capabilities g_device_caps[SYCL_MAX_DEVICES] = {
    {0, false, 0, -1}};

struct sycl_device_id2index {
  int index;
};

static void* g_scratch_buffer = nullptr;
static size_t g_scratch_size = 0; // disabled by default
static size_t g_scratch_offset = 0;

static dpct::queue_ptr g_sycl_handles[SYCL_MAX_DEVICES] = {nullptr};

int get_main_device();

[[noreturn]] static inline void bad_arch(const sycl::stream& stream_ct1) {
  stream_ct1 << "ERROR: ggml-sycl was compiled without support for the "
                "current GPU architecture.\n";
  // __trap();
  std::exit(1);

  (void)bad_arch; // suppress unused function warning
}

/*
device_index: device index from 0 to n (continue numbers).
    It is used for device select/set in SYCL backend internal data structure.
*/
void check_allow_gpu_index(const int device_index);

/*
device_id: device ID is shown by ggml_backend_sycl_print_sycl_devices().
    It is only used to set current working device.
*/
void check_allow_gpu_id(const int device_id);

int get_current_device_id();

inline dpct::err0 ggml_sycl_set_device(const int device) try {
  int device_id = g_sycl_gpu_mgr->gpus[device];
  check_allow_gpu_id(device_id);

  int current_device_id;
  SYCL_CHECK(CHECK_TRY_ERROR(current_device_id = get_current_device_id()));

  // GGML_SYCL_DEBUG("ggml_sycl_set_device device_id=%d,
  // current_device_id=%d\n", device, current_device);
  if (device_id == current_device_id) {
    return 0;
  }

  return CHECK_TRY_ERROR(dpct::select_device(device_id));
} catch (sycl::exception const& exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  crash();
  std::exit(1);
}

void log_ggml_var_device(
    const char* name,
    float* src,
    size_t total_elements,
    bool src_on_device);

void log_ggml_var_device_fp16(
    const char* name,
    sycl::half* src,
    size_t total_elements,
    bool src_on_device);

// todo: debug for crash in some case
void print_ggml_tensor(const char* name, struct ggml_tensor* src);

static int log_file_name_idx = 0;
void log_tensor_with_cnt(
    const char* name,
    struct ggml_tensor* src,
    int stop_cnt);

#endif // GGML_SYCL_COMMON_HPP