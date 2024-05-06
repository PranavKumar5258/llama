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

#ifndef GGML_SYCL_POOL_HPP
#define GGML_SYCL_POOL_HPP

// buffer pool for sycl
#define MAX_SYCL_BUFFERS 256

struct scoped_spin_lock {
    std::atomic_flag& lock;
    scoped_spin_lock(std::atomic_flag& lock) : lock(lock) {
        while (lock.test_and_set(std::memory_order_acquire)) {
            ; // spin
        }
    }
    ~scoped_spin_lock() {
        lock.clear(std::memory_order_release);
    }
    scoped_spin_lock(const scoped_spin_lock&) = delete;
    scoped_spin_lock& operator=(const scoped_spin_lock&) = delete;
};

static std::atomic_flag g_sycl_pool_lock = ATOMIC_FLAG_INIT;

// #define DEBUG_SYCL_MALLOC
struct sycl_buffer {
    void * ptr = nullptr;
    size_t size = 0;
};

static sycl_buffer g_sycl_buffer_pool[GGML_SYCL_MAX_DEVICES][MAX_SYCL_BUFFERS];
static size_t g_sycl_pool_size[GGML_SYCL_MAX_DEVICES] = {0};

static void *ggml_sycl_pool_malloc_leg(int device_index, size_t size, size_t *actual_size) try {
    scoped_spin_lock lock(g_sycl_pool_lock);
    // GGML_SYCL_DEBUG("ggml_sycl_pool_malloc_leg device_index %d size=%lu\n", device_index, size);
#ifdef DEBUG_SYCL_MALLOC
    int nnz = 0;
    size_t max_size = 0;
#endif
    size_t best_diff = 1ull << 36;
    int ibest = -1;
    for (int i = 0; i < MAX_SYCL_BUFFERS; ++i) {
        sycl_buffer& b = g_sycl_buffer_pool[device_index][i];
        if (b.ptr != nullptr) {
#ifdef DEBUG_SYCL_MALLOC
            ++nnz;
            if (b.size > max_size) max_size = b.size;
#endif
            if (b.size >= size) {
                size_t diff = b.size - size;
                if (diff < best_diff) {
                    best_diff = diff;
                    ibest = i;
                    if (!best_diff) {
                        void * ptr = b.ptr;
                        *actual_size = b.size;
                        b.ptr = nullptr;
                        b.size = 0;
                        // GGML_SYCL_DEBUG("ggml_sycl_pool_malloc_leg return 1 %p and rm in pool\n", ptr);
                        return ptr;
                    }
                }
            }
        }
    }
    if (ibest >= 0) {
        sycl_buffer& b = g_sycl_buffer_pool[device_index][ibest];
        void * ptr = b.ptr;
        *actual_size = b.size;
        b.ptr = nullptr;
        b.size = 0;
        // GGML_SYCL_DEBUG("ggml_sycl_pool_malloc_leg return 2 %p and rm in pool\n", ptr);
        return ptr;
    }
    void * ptr;
    size_t look_ahead_size = (size_t) (1.05 * size);
    look_ahead_size = 256 * ((look_ahead_size + 255)/256);

    const dpct::queue_ptr stream = g_syclStreams[device_index][0];
    SYCL_CHECK(
        CHECK_TRY_ERROR(ptr = (void *)sycl::malloc_device(
                             look_ahead_size, *stream)));
    *actual_size = look_ahead_size;
    g_sycl_pool_size[device_index] += look_ahead_size;

#ifdef DEBUG_SYCL_MALLOC
    fprintf(stderr, "%s[%d]: %d buffers, max_size = %u MB, pool_size = %u MB, requested %u MB\n", __func__, id, nnz,
            (uint32_t)(max_size/1024/1024), (uint32_t)(g_sycl_pool_size[id]/1024/1024), (uint32_t)(size/1024/1024));
#endif
    // GGML_SYCL_DEBUG("ggml_sycl_pool_malloc_leg look_ahead_size=%lu, return %p\n", look_ahead_size, ptr);
    return ptr;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

static void ggml_sycl_pool_free_leg(int device_index, void *ptr, size_t size) try {
    scoped_spin_lock lock(g_sycl_pool_lock);
    const dpct::queue_ptr stream = g_syclStreams[device_index][0];
    for (int i = 0; i < MAX_SYCL_BUFFERS; ++i) {
        sycl_buffer& b = g_sycl_buffer_pool[device_index][i];
        if (b.ptr == nullptr) {
            b.ptr = ptr;
            b.size = size;
            return;
        }
    }
    fprintf(stderr, "WARNING: sycl buffer pool full, increase MAX_SYCL_BUFFERS\n");
    SYCL_CHECK(CHECK_TRY_ERROR(sycl::free(ptr, *stream)));
    g_sycl_pool_size[device_index] -= size;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

// pool with virtual memory
/*
DPCT1082:64: Migration of CUmemGenericAllocationHandle type is not supported.
*/
// static std::vector<CUmemGenericAllocationHandle>
//     g_sycl_pool_handles[GGML_SYCL_MAX_DEVICES];
static dpct::device_ptr g_sycl_pool_addr[GGML_SYCL_MAX_DEVICES] = {0};
static size_t g_sycl_pool_used[GGML_SYCL_MAX_DEVICES] = {0};

static void *ggml_sycl_pool_malloc_vmm(int device_index, size_t size, size_t *actual_size) try {
    GGML_UNUSED(device_index);
    GGML_UNUSED(size);
    GGML_UNUSED(actual_size);
    return NULL;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

static void ggml_sycl_pool_free_vmm(int device_index, void *ptr, size_t size) try {
    scoped_spin_lock lock(g_sycl_pool_lock);
#ifdef DEBUG_SYCL_MALLOC
    printf("sycl pool[%d]: freed %llu bytes at %llx\n", device_index, (unsigned long long) size, ptr);
#endif

    g_sycl_pool_used[device_index] -= size;

    // all deallocations must be in reverse order of the allocations
    GGML_ASSERT(ptr == (void *) (g_sycl_pool_addr[device_index] + g_sycl_pool_used[device_index]));
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

static void *ggml_sycl_pool_malloc(int device_index, size_t size, size_t *actual_size) try {
    if (g_device_caps[device_index].vmm) {
        return ggml_sycl_pool_malloc_vmm(device_index, size, actual_size);
    } else {
        return ggml_sycl_pool_malloc_leg(device_index, size, actual_size);
    }
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

static void ggml_sycl_pool_free(int device_index, void *ptr, size_t size) try {
    if (g_device_caps[device_index].vmm) {
        ggml_sycl_pool_free_vmm(device_index, ptr, size);
    } else {
        ggml_sycl_pool_free_leg(device_index, ptr, size);
    }
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

template<typename T>
struct sycl_pool_alloc {
    int device_index = -1;
    int device_id = -1;
    T * ptr = nullptr;
    size_t actual_size = 0;

    // size is in number of elements
    T * alloc(size_t size) {
        GGML_ASSERT(ptr == nullptr);
        device_id = get_current_device_id();
        device_index = g_sycl_gpu_mgr->get_index(device_id);
        ptr = (T *) ggml_sycl_pool_malloc(device_index, size * sizeof(T), &this->actual_size);
        // GGML_SYCL_DEBUG("sycl_pool_alloc %lu return %p actual size=%lu\n", size * sizeof(T), ptr, this->actual_size);
        return ptr;
    }

    sycl_pool_alloc(size_t size) {
        alloc(size);
    }

    ~sycl_pool_alloc() {
        if (ptr != nullptr) {
            ggml_sycl_pool_free(device_index, ptr, actual_size);
        }
    }

    T * get() {
        return ptr;
    }

    sycl_pool_alloc() = default;
    sycl_pool_alloc(const sycl_pool_alloc &) = delete;
    sycl_pool_alloc(sycl_pool_alloc &&) = delete;
    sycl_pool_alloc& operator=(const sycl_pool_alloc &) = delete;
    sycl_pool_alloc& operator=(sycl_pool_alloc &&) = delete;
};

#endif // GGML_SYCL_POOL_HPP
