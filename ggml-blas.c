#include "ggml-blas.h"
#include "ggml-backend-impl.h"

#include <stdlib.h>

#if defined(GGML_USE_ACCELERATE)
#   include <Accelerate/Accelerate.h>
#elif defined(GGML_USE_OPENBLAS)
#   if defined(GGML_BLAS_USE_MKL)
#       include <mkl.h>
#   else
#       include <cblas.h>
#   endif
#endif

// helper function to determine if it is better to use BLAS or not
// for large matrices, BLAS is faster
static bool ggml_compute_forward_mul_mat_use_blas(const struct ggml_tensor * dst) {
    const struct ggml_tensor * src0 = dst->src[0];
    const struct ggml_tensor * src1 = dst->src[1];

    //const int64_t ne00 = src0->ne[0];
    //const int64_t ne01 = src0->ne[1];

    const int64_t ne10 = src1->ne[0];

    const int64_t ne0 = dst->ne[0];
    const int64_t ne1 = dst->ne[1];

    // NOTE: with GGML_OP_MUL_MAT_ID we don't want to go through the BLAS branch because it will dequantize (to_float)
    //       all the experts for each batch element and the processing would become incredibly slow
    // TODO: find the optimal values for these
    if (dst->op != GGML_OP_MUL_MAT_ID &&
        ggml_is_contiguous(src0) &&
        ggml_is_contiguous(src1) &&
      //src0->type == GGML_TYPE_F32 &&
        src1->type == GGML_TYPE_F32 &&
        (ne0 >= 32 && ne1 >= 32 && ne10 >= 32)) {

        /*printf("BLAS: %d %d %d %d %d\n", ne0, ne1, ne10, ne00, ne01);*/
        return true;
    }

    return false;
}


static void ggml_backend_blas_mul_mat(struct ggml_tensor * dst) {
    const struct ggml_tensor * src0 = dst->src[0];
    const struct ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_BINARY_OP_LOCALS

    // TODO: parallelize dequantize
    const int ith = 0;
    const int nth = 1;

    const enum ggml_type type = src0->type;

    ggml_type_traits_t type_traits = ggml_internal_get_type_traits(type);

    GGML_ASSERT(ne0 == ne01);
    GGML_ASSERT(ne1 == ne11);
    GGML_ASSERT(ne2 == ne12);
    GGML_ASSERT(ne3 == ne13);

    // we don't support permuted src0 or src1
    GGML_ASSERT(nb00 == ggml_type_size(type));
    GGML_ASSERT(nb10 == ggml_type_size(src1->type));

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 == sizeof(float));
    GGML_ASSERT(nb0 <= nb1);
    GGML_ASSERT(nb1 <= nb2);
    GGML_ASSERT(nb2 <= nb3);

    // broadcast factors
    const int64_t r2 = ne12/ne02;
    const int64_t r3 = ne13/ne03;

    // nb01 >= nb00 - src0 is not transposed
    //   compute by src0 rows

    const int64_t ne_plane      = ne01*ne00;
    const size_t  desired_wsize = ne13*ne12*ne_plane*sizeof(float);

    static void * wdata = NULL;
    wdata = realloc(wdata, desired_wsize);
    GGML_ASSERT(wdata != NULL);

    // convert src0 to float
    if (true) {
        if (type != GGML_TYPE_F32) {
            ggml_to_float_t  const to_float = type_traits.to_float;
            // TODO: parallelize by src0 rows
            for (int64_t i13 = 0; i13 < ne13; i13++) {
                for (int64_t i12 = 0; i12 < ne12; i12++) {
                    // broadcast src0 into src1 across 2nd,3rd dimension
                    const int64_t i03 = i13/r3;
                    const int64_t i02 = i12/r2;

                    const void  *       x      = (char *)  src0->data + i02*nb02          + i03*nb03;
                          float * const wplane = (float *) wdata      + i13*ne12*ne_plane + i12*ne_plane;

                    for (int64_t i01 = ith; i01 < ne01; i01 += nth) {
                        to_float((const char *) x + i01*nb01, wplane + i01*ne00, ne00);
                    }
                }
            }
        }
    }

    for (int64_t i13 = 0; i13 < ne13; i13++) {
        for (int64_t i12 = 0; i12 < ne12; i12++) {
            const int64_t i03 = i13/r3;
            const int64_t i02 = i12/r2;

            const void  * x = (char *)            src0->data + i02*nb02 + i03*nb03;
            const float * y = (float *) ((char *) src1->data + i12*nb12 + i13*nb13);
                    float * d = (float *) ((char *)  dst->data + i12*nb2  + i13*nb3);

            if (type != GGML_TYPE_F32) {
                x = (float *) wdata + i13*ne12*ne_plane + i12*ne_plane;
            }

            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                        ne1, ne01, ne10,
                        1.0f,   y, ne10,
                                x, ne00,
                        0.0f,   d, ne01);
        }
    }
}

// backend interface

GGML_CALL static const char * ggml_backend_blas_name(ggml_backend_t backend) {
    return "BLAS";

    GGML_UNUSED(backend);
}

GGML_CALL static void ggml_backend_blas_free(ggml_backend_t backend) {
    free(backend);
}

GGML_CALL static ggml_backend_buffer_type_t ggml_backend_blas_get_default_buffer_type(ggml_backend_t backend) {
    return ggml_backend_cpu_buffer_type();

    GGML_UNUSED(backend);
}

GGML_CALL static enum ggml_status ggml_backend_blas_graph_compute(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    for (int i = 0; i < cgraph->n_nodes; i++) {
        struct ggml_tensor * node = cgraph->nodes[i];

        switch (node->op) {
            case GGML_OP_MUL_MAT:
                ggml_backend_blas_mul_mat(node);
                break;

            // TODO
            //case GGML_OP_OUT_PROD:

            case GGML_OP_NONE:
            case GGML_OP_RESHAPE:
            case GGML_OP_VIEW:
            case GGML_OP_PERMUTE:
            case GGML_OP_TRANSPOSE:
                break;

            default:
                fprintf(stderr, "%s: unsupported op %s\n", __func__, ggml_op_desc(node));
                GGML_ASSERT(false);
        }
    }

    return GGML_STATUS_SUCCESS;

    GGML_UNUSED(backend);
}

GGML_CALL static bool ggml_backend_blas_supports_op(ggml_backend_t backend, const struct ggml_tensor * op) {
    return op->op == GGML_OP_MUL_MAT && ggml_compute_forward_mul_mat_use_blas(op);

    GGML_UNUSED(backend);
}

static struct ggml_backend_i blas_backend_i = {
    /* .get_name                = */ ggml_backend_blas_name,
    /* .free                    = */ ggml_backend_blas_free,
    /* .get_default_buffer_type = */ ggml_backend_blas_get_default_buffer_type,
    /* .set_tensor_async        = */ NULL,
    /* .get_tensor_async        = */ NULL,
    /* .cpy_tensor_async        = */ NULL,
    /* .synchronize             = */ NULL,
    /* .graph_plan_create       = */ NULL,
    /* .graph_plan_free         = */ NULL,
    /* .graph_plan_compute      = */ NULL,
    /* .graph_compute           = */ ggml_backend_blas_graph_compute,
    /* .supports_op             = */ ggml_backend_blas_supports_op,
    /* .offload_op              = */ NULL,
    /* .event_new               = */ NULL,
    /* .event_free              = */ NULL,
    /* .event_record            = */ NULL,
    /* .event_wait              = */ NULL,
    /* .event_synchronize       = */ NULL,
};

static ggml_guid_t ggml_backend_blas_guid(void) {
    static ggml_guid guid = { 0x12, 0xa8, 0xae, 0xf4, 0xc0, 0x1e, 0x61, 0x97, 0x8f, 0xeb, 0x33, 0x04, 0xa1, 0x33, 0x51, 0x2d };
    return &guid;
}

ggml_backend_t ggml_backend_blas_init(void) {
    ggml_backend_t backend = malloc(sizeof(struct ggml_backend));
    if (backend == NULL) {
        return NULL;
    }

    *backend = (struct ggml_backend) {
        /* .guid      = */ ggml_backend_blas_guid(),
        /* .interface = */ blas_backend_i,
        /* .context   = */ NULL
    };

    return backend;
}
