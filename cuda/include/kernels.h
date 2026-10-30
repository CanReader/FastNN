#ifndef FASTNN_KERNELS_H
#define FASTNN_KERNELS_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

int fastnn_cuda_init(int device_id);
int fastnn_cuda_device_count();
int fastnn_cuda_synchronize();
void fastnn_cuda_get_memory_info(size_t* free_mem, size_t* total_mem);

int fastnn_cuda_malloc(float** ptr, size_t size);
int fastnn_cuda_free(float* ptr);
int fastnn_cuda_memcpy_h2d(float* dst, const float* src, size_t size);
int fastnn_cuda_memcpy_d2h(float* dst, const float* src, size_t size);
int fastnn_cuda_memcpy_d2d(float* dst, const float* src, size_t size);
int fastnn_cuda_memset(float* ptr, int value, size_t size);

int fastnn_cuda_add(const float* a, const float* b, float* out, size_t n);
int fastnn_cuda_sub(const float* a, const float* b, float* out, size_t n);
int fastnn_cuda_mul(const float* a, const float* b, float* out, size_t n);
int fastnn_cuda_div(const float* a, const float* b, float* out, size_t n);
int fastnn_cuda_add_scalar(const float* a, float scalar, float* out, size_t n);
int fastnn_cuda_mul_scalar(const float* a, float scalar, float* out, size_t n);
int fastnn_cuda_pow_scalar(const float* a, float scalar, float* out, size_t n);
int fastnn_cuda_sqrt(const float* a, float* out, size_t n);
int fastnn_cuda_abs(const float* a, float* out, size_t n);
int fastnn_cuda_neg(const float* a, float* out, size_t n);
int fastnn_cuda_exp(const float* a, float* out, size_t n);
int fastnn_cuda_log(const float* a, float* out, size_t n);
int fastnn_cuda_clamp(const float* a, float min_val, float max_val, float* out, size_t n);

int fastnn_cuda_relu(const float* input, float* output, size_t n);
int fastnn_cuda_relu_backward(const float* grad_output, const float* input, float* grad_input, size_t n);
int fastnn_cuda_leaky_relu(const float* input, float negative_slope, float* output, size_t n);
int fastnn_cuda_sigmoid(const float* input, float* output, size_t n);
int fastnn_cuda_sigmoid_backward(const float* grad_output, const float* output, float* grad_input, size_t n);
int fastnn_cuda_tanh_forward(const float* input, float* output, size_t n);
int fastnn_cuda_tanh_backward(const float* grad_output, const float* output, float* grad_input, size_t n);
int fastnn_cuda_gelu(const float* input, float* output, size_t n);
int fastnn_cuda_gelu_backward(const float* grad_output, const float* input, float* grad_input, size_t n);
int fastnn_cuda_silu(const float* input, float* output, size_t n);
int fastnn_cuda_silu_backward(const float* grad_output, const float* input, float* grad_input, size_t n);

int fastnn_cuda_softmax(const float* input, float* output, int batch_size, int num_classes);
int fastnn_cuda_softmax_backward(const float* grad_output, const float* output, float* grad_input, int batch_size, int num_classes);
int fastnn_cuda_log_softmax(const float* input, float* output, int batch_size, int num_classes);

int fastnn_cuda_matmul(const float* a, const float* b, float* c,
                        int m, int n, int k,
                        int lda, int ldb, int ldc,
                        float alpha, float beta);
int fastnn_cuda_matmul_batched(const float* a, const float* b, float* c,
                                int m, int n, int k, int batch_size,
                                float alpha, float beta);
int fastnn_cuda_matmul_nt(const float* a, const float* b, float* c, int m, int n, int k);
int fastnn_cuda_matmul_tn(const float* a, const float* b, float* c, int m, int n, int k);
int fastnn_cuda_matmul_batched_nt(const float* a, const float* b, float* c, int m, int n, int k, int batch_size);
int fastnn_cuda_matmul_batched_tn(const float* a, const float* b, float* c, int m, int n, int k, int batch_size);
int fastnn_cuda_transpose(const float* input, float* output, int rows, int cols);
int fastnn_cuda_transpose_batched(const float* input, float* output, int batch_size, int rows, int cols);
int fastnn_cuda_permute_nd(const float* input, float* output, const int* out_strides, const int* in_strides, const int* perm, int ndim, int numel);

int fastnn_cuda_sum(const float* input, float* output, size_t n);
int fastnn_cuda_sum_axis(const float* input, float* output, int* shape, int ndim, int axis, int total_elements);
int fastnn_cuda_mean(const float* input, float* output, size_t n);
int fastnn_cuda_max(const float* input, float* output, size_t n);
int fastnn_cuda_min(const float* input, float* output, size_t n);

int fastnn_cuda_layer_norm_forward(
    const float* input, const float* gamma, const float* beta,
    float* output, float* mean, float* inv_var,
    int batch_size, int normalized_size, float epsilon);

int fastnn_cuda_layer_norm_backward(
    const float* grad_output, const float* input,
    const float* gamma, const float* mean, const float* inv_var,
    float* grad_input, float* grad_gamma, float* grad_beta,
    int batch_size, int normalized_size);

int fastnn_cuda_embedding_forward(const int* indices, const float* weight, float* output, int num_indices, int embedding_dim);
int fastnn_cuda_embedding_backward(const int* indices, const float* grad_output, float* grad_weight, int num_indices, int embedding_dim, int num_embeddings);

#ifdef __cplusplus
}
#endif

#endif // FASTNN_KERNELS_H
