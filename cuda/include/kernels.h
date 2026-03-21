#ifndef FASTNN_KERNELS_H
#define FASTNN_KERNELS_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// Device Management
// ============================================================================
int fastnn_cuda_init(int device_id);
int fastnn_cuda_device_count();
int fastnn_cuda_synchronize();
void fastnn_cuda_get_memory_info(size_t* free_mem, size_t* total_mem);

// ============================================================================
// Memory Management
// ============================================================================
int fastnn_cuda_malloc(float** ptr, size_t size);
int fastnn_cuda_free(float* ptr);
int fastnn_cuda_memcpy_h2d(float* dst, const float* src, size_t size);
int fastnn_cuda_memcpy_d2h(float* dst, const float* src, size_t size);
int fastnn_cuda_memcpy_d2d(float* dst, const float* src, size_t size);
int fastnn_cuda_memset(float* ptr, int value, size_t size);

// ============================================================================
// Element-wise Operations
// ============================================================================
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

// ============================================================================
// Activation Functions
// ============================================================================
int fastnn_cuda_relu(const float* input, float* output, size_t n);
int fastnn_cuda_relu_backward(const float* grad_output, const float* input, float* grad_input, size_t n);
int fastnn_cuda_leaky_relu(const float* input, float negative_slope, float* output, size_t n);
int fastnn_cuda_leaky_relu_backward(const float* grad_output, const float* input, float negative_slope, float* grad_input, size_t n);
int fastnn_cuda_sigmoid(const float* input, float* output, size_t n);
int fastnn_cuda_sigmoid_backward(const float* grad_output, const float* output, float* grad_input, size_t n);
int fastnn_cuda_tanh_forward(const float* input, float* output, size_t n);
int fastnn_cuda_tanh_backward(const float* grad_output, const float* output, float* grad_input, size_t n);
int fastnn_cuda_gelu(const float* input, float* output, size_t n);
int fastnn_cuda_gelu_backward(const float* grad_output, const float* input, float* grad_input, size_t n);
int fastnn_cuda_silu(const float* input, float* output, size_t n);
int fastnn_cuda_silu_backward(const float* grad_output, const float* input, float* grad_input, size_t n);

// ============================================================================
// Softmax
// ============================================================================
int fastnn_cuda_softmax(const float* input, float* output, int batch_size, int num_classes);
int fastnn_cuda_softmax_backward(const float* grad_output, const float* output, float* grad_input, int batch_size, int num_classes);
int fastnn_cuda_log_softmax(const float* input, float* output, int batch_size, int num_classes);

// ============================================================================
// Matrix Operations (using custom kernels + cuBLAS)
// ============================================================================
int fastnn_cuda_matmul(const float* a, const float* b, float* c,
                        int m, int n, int k,
                        int lda, int ldb, int ldc,
                        float alpha, float beta);
int fastnn_cuda_matmul_batched(const float* a, const float* b, float* c,
                                int m, int n, int k, int batch_size,
                                float alpha, float beta);
int fastnn_cuda_transpose(const float* input, float* output, int rows, int cols);
int fastnn_cuda_transpose_batched(const float* input, float* output, int batch_size, int rows, int cols);

// ============================================================================
// Reduction Operations
// ============================================================================
int fastnn_cuda_sum(const float* input, float* output, size_t n);
int fastnn_cuda_sum_axis(const float* input, float* output, int* shape, int ndim, int axis, int total_elements);
int fastnn_cuda_mean(const float* input, float* output, size_t n);
int fastnn_cuda_max(const float* input, float* output, size_t n);
int fastnn_cuda_min(const float* input, float* output, size_t n);
int fastnn_cuda_argmax(const float* input, int* output, size_t n);
int fastnn_cuda_argmax_axis(const float* input, int* output, int outer, int axis_size, int inner);

// ============================================================================
// Convolution Operations
// ============================================================================
int fastnn_cuda_conv2d_forward(
    const float* input, const float* weight, const float* bias, float* output,
    int batch_size, int in_channels, int out_channels,
    int input_h, int input_w,
    int kernel_h, int kernel_w,
    int stride_h, int stride_w,
    int pad_h, int pad_w);

int fastnn_cuda_conv2d_backward_data(
    const float* grad_output, const float* weight, float* grad_input,
    int batch_size, int in_channels, int out_channels,
    int input_h, int input_w, int output_h, int output_w,
    int kernel_h, int kernel_w,
    int stride_h, int stride_w,
    int pad_h, int pad_w);

int fastnn_cuda_conv2d_backward_weight(
    const float* input, const float* grad_output, float* grad_weight,
    int batch_size, int in_channels, int out_channels,
    int input_h, int input_w, int output_h, int output_w,
    int kernel_h, int kernel_w,
    int stride_h, int stride_w,
    int pad_h, int pad_w);

// ============================================================================
// Pooling Operations
// ============================================================================
int fastnn_cuda_max_pool2d(
    const float* input, float* output, int* indices,
    int batch_size, int channels,
    int input_h, int input_w,
    int kernel_h, int kernel_w,
    int stride_h, int stride_w,
    int pad_h, int pad_w);

int fastnn_cuda_max_pool2d_backward(
    const float* grad_output, const int* indices, float* grad_input,
    int batch_size, int channels,
    int input_h, int input_w,
    int output_h, int output_w);

int fastnn_cuda_avg_pool2d(
    const float* input, float* output,
    int batch_size, int channels,
    int input_h, int input_w,
    int kernel_h, int kernel_w,
    int stride_h, int stride_w,
    int pad_h, int pad_w);

int fastnn_cuda_adaptive_avg_pool2d(
    const float* input, float* output,
    int batch_size, int channels,
    int input_h, int input_w,
    int output_h, int output_w);

// ============================================================================
// Normalization
// ============================================================================
int fastnn_cuda_batch_norm_forward(
    const float* input, const float* gamma, const float* beta,
    float* running_mean, float* running_var,
    float* output, float* save_mean, float* save_inv_var,
    int batch_size, int channels, int spatial_size,
    float momentum, float epsilon, int training);

int fastnn_cuda_batch_norm_backward(
    const float* grad_output, const float* input,
    const float* gamma, const float* save_mean, const float* save_inv_var,
    float* grad_input, float* grad_gamma, float* grad_beta,
    int batch_size, int channels, int spatial_size);

int fastnn_cuda_layer_norm_forward(
    const float* input, const float* gamma, const float* beta,
    float* output, float* mean, float* inv_var,
    int batch_size, int normalized_size, float epsilon);

int fastnn_cuda_layer_norm_backward(
    const float* grad_output, const float* input,
    const float* gamma, const float* mean, const float* inv_var,
    float* grad_input, float* grad_gamma, float* grad_beta,
    int batch_size, int normalized_size);

int fastnn_cuda_rms_norm_forward(
    const float* input, const float* gamma,
    float* output, float* rms,
    int batch_size, int normalized_size, float epsilon);

// ============================================================================
// Dropout
// ============================================================================
int fastnn_cuda_dropout_forward(const float* input, float* output, float* mask, float p, size_t n, unsigned long long seed);
int fastnn_cuda_dropout_backward(const float* grad_output, const float* mask, float* grad_input, float p, size_t n);

// ============================================================================
// Embedding
// ============================================================================
int fastnn_cuda_embedding_forward(const int* indices, const float* weight, float* output, int num_indices, int embedding_dim);
int fastnn_cuda_embedding_backward(const int* indices, const float* grad_output, float* grad_weight, int num_indices, int embedding_dim, int num_embeddings);

// ============================================================================
// Loss Functions
// ============================================================================
int fastnn_cuda_cross_entropy_loss(const float* log_probs, const int* targets, float* loss, float* grad, int batch_size, int num_classes);
int fastnn_cuda_mse_loss(const float* predictions, const float* targets, float* loss, float* grad, size_t n);
int fastnn_cuda_binary_cross_entropy(const float* predictions, const float* targets, float* loss, float* grad, size_t n);

// ============================================================================
// Attention / Transformer Operations
// ============================================================================
int fastnn_cuda_scaled_dot_product_attention(
    const float* query, const float* key, const float* value,
    float* output, float* attn_weights,
    const float* mask,
    int batch_size, int num_heads, int seq_len_q, int seq_len_k, int head_dim,
    float scale, int causal);

// ============================================================================
// Optimizer Kernels
// ============================================================================
int fastnn_cuda_sgd_step(float* params, const float* grads, float* velocity,
                          float lr, float momentum, float weight_decay, float dampening,
                          int nesterov, size_t n);

int fastnn_cuda_adam_step(float* params, const float* grads,
                           float* m, float* v,
                           float lr, float beta1, float beta2, float epsilon,
                           float weight_decay, int step, int amsgrad, float* v_max,
                           size_t n);

// ============================================================================
// Utility Kernels
// ============================================================================
int fastnn_cuda_fill(float* data, float value, size_t n);
int fastnn_cuda_copy(const float* src, float* dst, size_t n);
int fastnn_cuda_arange(float* output, float start, float step, size_t n);
int fastnn_cuda_where(const float* condition, const float* x, const float* y, float* output, size_t n);
int fastnn_cuda_gather(const float* input, const int* indices, float* output,
                        int outer_size, int gather_dim_size, int inner_size, int num_indices);

#ifdef __cplusplus
}
#endif

#endif // FASTNN_KERNELS_H
