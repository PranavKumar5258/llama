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

#include "common.hpp"

int get_main_device() {
  return g_main_device;
}

void check_allow_gpu_index(const int device_index) {
  if (device_index >= g_device_count) {
    char error_buf[256];
    snprintf(
        error_buf,
        sizeof(error_buf),
        "%s error: device_index:%d is out of range: [0-%d]",
        __func__,
        device_index,
        g_device_count - 1);
    fprintf(stderr, "%s\n", error_buf);
    assert(false);
  }
}

int get_current_device_id() {
  return dpct::dev_mgr::instance().current_device_id();
}

void log_ggml_var_device(
    const char* name,
    float* src,
    size_t total_elements,
    bool src_on_device) {
  if (!g_ggml_sycl_debug)
    return;
  if (!src) {
    printf("GGML Tensor:%s skip to save for NULL pointer\n", name);
    return;
  }
  char filename[1024];
  sprintf(filename, "%s.txt", name);
  printf("GGML Tensor:%s save to %s\n", name, filename);

  size_t total_size = total_elements * sizeof(float);
  float* local_buf = NULL;
  if (src_on_device) {
    local_buf = (float*)ggml_sycl_host_malloc(total_size);
    ggml_sycl_set_device(g_main_device);
    dpct::queue_ptr main_stream = g_syclStreams[g_main_device][0];
    main_stream->memcpy(local_buf, src, total_size).wait();
  } else {
    local_buf = (float*)src;
  }

  std::ofstream logfile;
  logfile.open(filename);
  for (size_t i = 0; i < total_elements; i++) {
    logfile << local_buf[i] << " ";
    if ((i + 1) % 20 == 0)
      logfile << std::endl;
  }
  logfile << std::endl;
  logfile.close();

  if (src_on_device)
    ggml_sycl_host_free(local_buf);
}

void log_ggml_var_device_fp16(
    const char* name,
    sycl::half* src,
    size_t total_elements,
    bool src_on_device) {
  if (!g_ggml_sycl_debug)
    return;
  if (!src) {
    printf("GGML Tensor:%s skip to save for NULL pointer\n", name);
    return;
  }
  char filename[1024];
  sprintf(filename, "%s.txt", name);
  printf("GGML Tensor:%s save to %s\n", name, filename);

  size_t total_size = total_elements * sizeof(sycl::half);
  sycl::half* local_buf = NULL;
  if (src_on_device) {
    local_buf = (sycl::half*)ggml_sycl_host_malloc(total_size);
    ggml_sycl_set_device(g_main_device);
    dpct::queue_ptr main_stream = g_syclStreams[g_main_device][0];
    main_stream->memcpy(local_buf, src, total_size).wait();
  } else {
    local_buf = (sycl::half*)src;
  }

  std::ofstream logfile;
  logfile.open(filename);
  for (size_t i = 0; i < total_elements; i++) {
    logfile << local_buf[i] << " ";
    if ((i + 1) % 20 == 0)
      logfile << std::endl;
  }
  logfile << std::endl;
  logfile.close();

  if (src_on_device)
    ggml_sycl_host_free(local_buf);
}

void print_ggml_tensor(const char* name, struct ggml_tensor* src) {
  if (!g_ggml_sycl_debug)
    return;
  if (!src) {
    printf("GGML Tensor:%s skip to save for NULL pointer\n", name);
    return;
  }

  size_t total_elements = ggml_nelements(src);

  const bool src_on_device = src->backend == GGML_BACKEND_TYPE_GPU ||
      src->backend == GGML_BACKEND_TYPE_GPU_SPLIT;
  float* src_data = NULL;
  if (src_on_device) {
    ggml_tensor_extra_gpu* src_extra = (ggml_tensor_extra_gpu*)src->extra;
    src_data = (float*)src_extra->data_device[g_main_device];
  } else {
    src_data = (float*)src->data;
  }

  log_ggml_var_device(name, src_data, total_elements, src_on_device);
}

void log_tensor_with_cnt(
    const char* name,
    struct ggml_tensor* src,
    int stop_cnt) {
  stop_cnt = 4;
  if (log_file_name_idx >= stop_cnt)
    return;
  char filename[1280];
  sprintf(filename, "%s_%07d", name, log_file_name_idx);
  log_file_name_idx++;
  print_ggml_tensor(filename, src);
}

void* ggml_sycl_host_malloc(size_t size) try {
  if (getenv("GGML_SYCL_NO_PINNED") != nullptr) {
    return nullptr;
  }

  void* ptr = nullptr;
  // allow to use dpct::get_in_order_queue() for host malloc
  dpct::err0 err = CHECK_TRY_ERROR(
      ptr = (void*)sycl::malloc_host(size, dpct::get_in_order_queue()));

  if (err != 0) {
    // clear the error
    fprintf(
        stderr,
        "WARNING: failed to allocate %.2f MB of pinned memory: %s\n",
        size / 1024.0 / 1024.0,
        "syclGetErrorString is not supported");
    return nullptr;
  }

  return ptr;
} catch (sycl::exception const& exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

void ggml_sycl_host_free(void* ptr) try {
  // allow to use dpct::get_in_order_queue() for host malloc
  SYCL_CHECK(CHECK_TRY_ERROR(sycl::free(ptr, dpct::get_in_order_queue())));
} catch (sycl::exception const& exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}
