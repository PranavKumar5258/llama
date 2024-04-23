#pragma once
#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#ifdef __cplusplus
extern "C" {
#endif

struct ggml_tensor;
struct ggml_compute_params;

bool llamafile_sgemm(intptr_t, intptr_t, intptr_t, const void *, intptr_t,
                     const void *, intptr_t, void *, intptr_t, int, int,
                     int, int, int, int);

bool llamafile_mixmul(const struct ggml_compute_params *, const struct ggml_tensor *,
                      const struct ggml_tensor *, const struct ggml_tensor *,
                      struct ggml_tensor *);

size_t llamafile_mixmul_needs(const struct ggml_tensor *,
                              const struct ggml_tensor *,
                              const struct ggml_tensor *);

#ifdef __cplusplus
}
#endif
