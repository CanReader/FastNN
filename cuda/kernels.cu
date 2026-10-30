#include "include/kernels.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <curand.h>
#include <curand_kernel.h>
#include <math.h>
#include <float.h>
#include <stdio.h>

#define BLOCK_SIZE 256
#define TILE_SIZE 32
#define WARP_SIZE 32
#define PERMUTE_MAX_DIMS 8

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) return -1; \
} while(0)

#define CUBLAS_CHECK(call) do { \
    cublasStatus_t stat = (call); \
    if (stat != CUBLAS_STATUS_SUCCESS) return -1; \
} while(0)

static cublasHandle_t g_cublas_handle = nullptr;
static curandGenerator_t g_curand_gen = nullptr;

static inline int div_ceil(int a, int b) { return (a + b - 1) / b; }

extern "C" int fastnn_cuda_init(int device_id) {
    CUDA_CHECK(cudaSetDevice(device_id));

    if (g_cublas_handle == nullptr) {
        CUBLAS_CHECK(cublasCreate(&g_cublas_handle));
        CUBLAS_CHECK(cublasSetMathMode(g_cublas_handle, CUBLAS_TF32_TENSOR_OP_MATH));
    }

    if (g_curand_gen == nullptr) {
        curandCreateGenerator(&g_curand_gen, CURAND_RNG_PSEUDO_DEFAULT);
        curandSetPseudoRandomGeneratorSeed(g_curand_gen, 42);
    }

    return 0;
}

extern "C" int fastnn_cuda_device_count() {
    int count = 0;
    cudaGetDeviceCount(&count);
    return count;
}

extern "C" int fastnn_cuda_synchronize() {
    CUDA_CHECK(cudaDeviceSynchronize());
    return 0;
}

extern "C" void fastnn_cuda_get_memory_info(size_t* free_mem, size_t* total_mem) {
    cudaMemGetInfo(free_mem, total_mem);
}

extern "C" int fastnn_cuda_malloc(float** ptr, size_t size) {
    CUDA_CHECK(cudaMalloc(ptr, size));
    return 0;
}

extern "C" int fastnn_cuda_free(float* ptr) {
    CUDA_CHECK(cudaFree(ptr));
    return 0;
}

extern "C" int fastnn_cuda_memcpy_h2d(float* dst, const float* src, size_t size) {
    CUDA_CHECK(cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice));
    return 0;
}

extern "C" int fastnn_cuda_memcpy_d2h(float* dst, const float* src, size_t size) {
    CUDA_CHECK(cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost));
    return 0;
}

extern "C" int fastnn_cuda_memcpy_d2d(float* dst, const float* src, size_t size) {
    CUDA_CHECK(cudaMemcpy(dst, src, size, cudaMemcpyDeviceToDevice));
    return 0;
}

extern "C" int fastnn_cuda_memset(float* ptr, int value, size_t size) {
    CUDA_CHECK(cudaMemset(ptr, value, size));
    return 0;
}

__global__ void kernel_add(const float* a, const float* b, float* out, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = a[idx] + b[idx];
}

__global__ void kernel_sub(const float* a, const float* b, float* out, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = a[idx] - b[idx];
}

__global__ void kernel_mul(const float* a, const float* b, float* out, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = a[idx] * b[idx];
}

__global__ void kernel_div(const float* a, const float* b, float* out, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = a[idx] / b[idx];
}

__global__ void kernel_add_scalar(const float* a, float scalar, float* out, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = a[idx] + scalar;
}

__global__ void kernel_mul_scalar(const float* a, float scalar, float* out, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = a[idx] * scalar;
}

__global__ void kernel_pow_scalar(const float* a, float scalar, float* out, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = powf(a[idx], scalar);
}

__global__ void kernel_sqrt(const float* a, float* out, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = sqrtf(a[idx]);
}

__global__ void kernel_abs(const float* a, float* out, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = fabsf(a[idx]);
}

__global__ void kernel_neg(const float* a, float* out, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = -a[idx];
}

__global__ void kernel_exp(const float* a, float* out, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = expf(a[idx]);
}

__global__ void kernel_log(const float* a, float* out, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = logf(a[idx]);
}

__global__ void kernel_clamp(const float* a, float min_val, float max_val, float* out, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = fminf(fmaxf(a[idx], min_val), max_val);
}

// The tree reductions halve blockDim.x each round, so a non-power-of-two thread
// count drops the tail (5 threads: stride 2 folds 2->0 and 3->1, thread 4 is
// never read). Round up; extra threads skip their loop and add the identity.
__host__ inline int reduction_threads(int n) {
    int threads = 1;
    while (threads < n && threads < BLOCK_SIZE) threads <<= 1;
    return threads;
}

#define LAUNCH_ELEMENTWISE(kernel, ...) \
    do { \
        int blocks = div_ceil((int)n, BLOCK_SIZE); \
        kernel<<<blocks, BLOCK_SIZE>>>(__VA_ARGS__); \
        CUDA_CHECK(cudaGetLastError()); \
        return 0; \
    } while(0)

extern "C" int fastnn_cuda_add(const float* a, const float* b, float* out, size_t n) { LAUNCH_ELEMENTWISE(kernel_add, a, b, out, n); }
extern "C" int fastnn_cuda_sub(const float* a, const float* b, float* out, size_t n) { LAUNCH_ELEMENTWISE(kernel_sub, a, b, out, n); }
extern "C" int fastnn_cuda_mul(const float* a, const float* b, float* out, size_t n) { LAUNCH_ELEMENTWISE(kernel_mul, a, b, out, n); }
extern "C" int fastnn_cuda_div(const float* a, const float* b, float* out, size_t n) { LAUNCH_ELEMENTWISE(kernel_div, a, b, out, n); }
extern "C" int fastnn_cuda_add_scalar(const float* a, float s, float* out, size_t n) { LAUNCH_ELEMENTWISE(kernel_add_scalar, a, s, out, n); }
extern "C" int fastnn_cuda_mul_scalar(const float* a, float s, float* out, size_t n) { LAUNCH_ELEMENTWISE(kernel_mul_scalar, a, s, out, n); }
extern "C" int fastnn_cuda_pow_scalar(const float* a, float s, float* out, size_t n) { LAUNCH_ELEMENTWISE(kernel_pow_scalar, a, s, out, n); }
extern "C" int fastnn_cuda_sqrt(const float* a, float* out, size_t n) { LAUNCH_ELEMENTWISE(kernel_sqrt, a, out, n); }
extern "C" int fastnn_cuda_abs(const float* a, float* out, size_t n) { LAUNCH_ELEMENTWISE(kernel_abs, a, out, n); }
extern "C" int fastnn_cuda_neg(const float* a, float* out, size_t n) { LAUNCH_ELEMENTWISE(kernel_neg, a, out, n); }
extern "C" int fastnn_cuda_exp(const float* a, float* out, size_t n) { LAUNCH_ELEMENTWISE(kernel_exp, a, out, n); }
extern "C" int fastnn_cuda_log(const float* a, float* out, size_t n) { LAUNCH_ELEMENTWISE(kernel_log, a, out, n); }
extern "C" int fastnn_cuda_clamp(const float* a, float min_v, float max_v, float* out, size_t n) { LAUNCH_ELEMENTWISE(kernel_clamp, a, min_v, max_v, out, n); }

__global__ void kernel_relu(const float* input, float* output, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) output[idx] = fmaxf(input[idx], 0.0f);
}

__global__ void kernel_relu_backward(const float* grad_output, const float* input, float* grad_input, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) grad_input[idx] = input[idx] > 0.0f ? grad_output[idx] : 0.0f;
}

__global__ void kernel_leaky_relu(const float* input, float neg_slope, float* output, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = input[idx];
        output[idx] = val > 0.0f ? val : neg_slope * val;
    }
}

__global__ void kernel_sigmoid(const float* input, float* output, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) output[idx] = 1.0f / (1.0f + expf(-input[idx]));
}

__global__ void kernel_sigmoid_backward(const float* grad_output, const float* output, float* grad_input, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) grad_input[idx] = grad_output[idx] * output[idx] * (1.0f - output[idx]);
}

__global__ void kernel_tanh_forward(const float* input, float* output, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) output[idx] = tanhf(input[idx]);
}

__global__ void kernel_tanh_backward(const float* grad_output, const float* output, float* grad_input, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) grad_input[idx] = grad_output[idx] * (1.0f - output[idx] * output[idx]);
}

__global__ void kernel_gelu(const float* input, float* output, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = input[idx];
        output[idx] = 0.5f * x * (1.0f + erff(x * 0.7071067811865475f));
    }
}

__global__ void kernel_gelu_backward(const float* grad_output, const float* input, float* grad_input, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = input[idx];
        float cdf = 0.5f * (1.0f + erff(x * 0.7071067811865475f));
        float pdf = expf(-0.5f * x * x) * 0.3989422804014327f; // 1/sqrt(2*pi)
        grad_input[idx] = grad_output[idx] * (cdf + x * pdf);
    }
}

__global__ void kernel_silu(const float* input, float* output, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = input[idx];
        float sig = 1.0f / (1.0f + expf(-x));
        output[idx] = x * sig;
    }
}

__global__ void kernel_silu_backward(const float* grad_output, const float* input, float* grad_input, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = input[idx];
        float sig = 1.0f / (1.0f + expf(-x));
        grad_input[idx] = grad_output[idx] * (sig + x * sig * (1.0f - sig));
    }
}

extern "C" int fastnn_cuda_relu(const float* in, float* out, size_t n) { LAUNCH_ELEMENTWISE(kernel_relu, in, out, n); }
extern "C" int fastnn_cuda_relu_backward(const float* go, const float* in, float* gi, size_t n) { LAUNCH_ELEMENTWISE(kernel_relu_backward, go, in, gi, n); }
extern "C" int fastnn_cuda_leaky_relu(const float* in, float ns, float* out, size_t n) { LAUNCH_ELEMENTWISE(kernel_leaky_relu, in, ns, out, n); }
extern "C" int fastnn_cuda_sigmoid(const float* in, float* out, size_t n) { LAUNCH_ELEMENTWISE(kernel_sigmoid, in, out, n); }
extern "C" int fastnn_cuda_sigmoid_backward(const float* go, const float* out, float* gi, size_t n) { LAUNCH_ELEMENTWISE(kernel_sigmoid_backward, go, out, gi, n); }
extern "C" int fastnn_cuda_tanh_forward(const float* in, float* out, size_t n) { LAUNCH_ELEMENTWISE(kernel_tanh_forward, in, out, n); }
extern "C" int fastnn_cuda_tanh_backward(const float* go, const float* out, float* gi, size_t n) { LAUNCH_ELEMENTWISE(kernel_tanh_backward, go, out, gi, n); }
extern "C" int fastnn_cuda_gelu(const float* in, float* out, size_t n) { LAUNCH_ELEMENTWISE(kernel_gelu, in, out, n); }
extern "C" int fastnn_cuda_gelu_backward(const float* go, const float* in, float* gi, size_t n) { LAUNCH_ELEMENTWISE(kernel_gelu_backward, go, in, gi, n); }
extern "C" int fastnn_cuda_silu(const float* in, float* out, size_t n) { LAUNCH_ELEMENTWISE(kernel_silu, in, out, n); }
extern "C" int fastnn_cuda_silu_backward(const float* go, const float* in, float* gi, size_t n) { LAUNCH_ELEMENTWISE(kernel_silu_backward, go, in, gi, n); }

__global__ void kernel_softmax(const float* input, float* output, int batch_size, int num_classes) {
    int batch_idx = blockIdx.x;
    if (batch_idx >= batch_size) return;

    extern __shared__ float shared[];
    const float* in_row = input + batch_idx * num_classes;
    float* out_row = output + batch_idx * num_classes;

    // subtract the row max so expf cannot overflow
    float thread_max = -FLT_MAX;
    for (int i = threadIdx.x; i < num_classes; i += blockDim.x) {
        thread_max = fmaxf(thread_max, in_row[i]);
    }
    shared[threadIdx.x] = thread_max;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride)
            shared[threadIdx.x] = fmaxf(shared[threadIdx.x], shared[threadIdx.x + stride]);
        __syncthreads();
    }
    float max_val = shared[0];
    __syncthreads();

    float thread_sum = 0.0f;
    for (int i = threadIdx.x; i < num_classes; i += blockDim.x) {
        float val = expf(in_row[i] - max_val);
        out_row[i] = val;
        thread_sum += val;
    }
    shared[threadIdx.x] = thread_sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride)
            shared[threadIdx.x] += shared[threadIdx.x + stride];
        __syncthreads();
    }
    float sum_val = shared[0];
    __syncthreads();

    for (int i = threadIdx.x; i < num_classes; i += blockDim.x) {
        out_row[i] /= sum_val;
    }
}

__global__ void kernel_softmax_backward(const float* grad_output, const float* output, float* grad_input,
                                         int batch_size, int num_classes) {
    int batch_idx = blockIdx.x;
    if (batch_idx >= batch_size) return;

    extern __shared__ float shared[];
    const float* go_row = grad_output + batch_idx * num_classes;
    const float* out_row = output + batch_idx * num_classes;
    float* gi_row = grad_input + batch_idx * num_classes;

    float thread_dot = 0.0f;
    for (int i = threadIdx.x; i < num_classes; i += blockDim.x) {
        thread_dot += go_row[i] * out_row[i];
    }
    shared[threadIdx.x] = thread_dot;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride)
            shared[threadIdx.x] += shared[threadIdx.x + stride];
        __syncthreads();
    }
    float dot = shared[0];
    __syncthreads();

    for (int i = threadIdx.x; i < num_classes; i += blockDim.x) {
        gi_row[i] = out_row[i] * (go_row[i] - dot);
    }
}

__global__ void kernel_log_softmax(const float* input, float* output, int batch_size, int num_classes) {
    int batch_idx = blockIdx.x;
    if (batch_idx >= batch_size) return;

    extern __shared__ float shared[];
    const float* in_row = input + batch_idx * num_classes;
    float* out_row = output + batch_idx * num_classes;

    float thread_max = -FLT_MAX;
    for (int i = threadIdx.x; i < num_classes; i += blockDim.x)
        thread_max = fmaxf(thread_max, in_row[i]);
    shared[threadIdx.x] = thread_max;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) shared[threadIdx.x] = fmaxf(shared[threadIdx.x], shared[threadIdx.x + s]);
        __syncthreads();
    }
    float max_val = shared[0];
    __syncthreads();

    float thread_sum = 0.0f;
    for (int i = threadIdx.x; i < num_classes; i += blockDim.x)
        thread_sum += expf(in_row[i] - max_val);
    shared[threadIdx.x] = thread_sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) shared[threadIdx.x] += shared[threadIdx.x + s];
        __syncthreads();
    }
    float log_sum = logf(shared[0]);
    __syncthreads();

    for (int i = threadIdx.x; i < num_classes; i += blockDim.x)
        out_row[i] = in_row[i] - max_val - log_sum;
}

extern "C" int fastnn_cuda_softmax(const float* input, float* output, int bs, int nc) {
    int threads = reduction_threads(nc);
    kernel_softmax<<<bs, threads, threads * sizeof(float)>>>(input, output, bs, nc);
    CUDA_CHECK(cudaGetLastError());
    return 0;
}

extern "C" int fastnn_cuda_softmax_backward(const float* go, const float* out, float* gi, int bs, int nc) {
    int threads = reduction_threads(nc);
    kernel_softmax_backward<<<bs, threads, threads * sizeof(float)>>>(go, out, gi, bs, nc);
    CUDA_CHECK(cudaGetLastError());
    return 0;
}

extern "C" int fastnn_cuda_log_softmax(const float* input, float* output, int bs, int nc) {
    int threads = reduction_threads(nc);
    kernel_log_softmax<<<bs, threads, threads * sizeof(float)>>>(input, output, bs, nc);
    CUDA_CHECK(cudaGetLastError());
    return 0;
}

extern "C" int fastnn_cuda_matmul(const float* a, const float* b, float* c,
                                   int m, int n, int k,
                                   int lda, int ldb, int ldc,
                                   float alpha, float beta) {
    // cuBLAS is column-major, so compute C^T = B^T * A^T
    CUBLAS_CHECK(cublasSgemm(g_cublas_handle,
                             CUBLAS_OP_N, CUBLAS_OP_N,
                             n, m, k,
                             &alpha,
                             b, ldb,
                             a, lda,
                             &beta,
                             c, ldc));
    return 0;
}

// C (m×n) = A (m×k) × B^T, B stored (n×k)
extern "C" int fastnn_cuda_matmul_nt(const float* a, const float* b, float* c,
                                      int m, int n, int k) {
    float alpha = 1.0f, beta = 0.0f;
    CUBLAS_CHECK(cublasSgemm(g_cublas_handle,
                             CUBLAS_OP_T, CUBLAS_OP_N,
                             n, m, k,
                             &alpha, b, k, a, k, &beta, c, n));
    return 0;
}

// C (m×n) = A^T × B (k×n), A stored (k×m)
extern "C" int fastnn_cuda_matmul_tn(const float* a, const float* b, float* c,
                                      int m, int n, int k) {
    float alpha = 1.0f, beta = 0.0f;
    CUBLAS_CHECK(cublasSgemm(g_cublas_handle,
                             CUBLAS_OP_N, CUBLAS_OP_T,
                             n, m, k,
                             &alpha, b, n, a, m, &beta, c, n));
    return 0;
}

// Batched C = A × B^T, B stored (batch×n×k)
extern "C" int fastnn_cuda_matmul_batched_nt(const float* a, const float* b, float* c,
                                               int m, int n, int k, int batch_size) {
    float alpha = 1.0f, beta = 0.0f;
    long long stride_a = (long long)m * k;
    long long stride_b = (long long)n * k;
    long long stride_c = (long long)m * n;
    CUBLAS_CHECK(cublasSgemmStridedBatched(g_cublas_handle,
                                            CUBLAS_OP_T, CUBLAS_OP_N,
                                            n, m, k,
                                            &alpha, b, k, stride_b,
                                            a, k, stride_a,
                                            &beta, c, n, stride_c,
                                            batch_size));
    return 0;
}

// Batched C = A^T × B, A stored (batch×k×m)
extern "C" int fastnn_cuda_matmul_batched_tn(const float* a, const float* b, float* c,
                                               int m, int n, int k, int batch_size) {
    float alpha = 1.0f, beta = 0.0f;
    long long stride_a = (long long)k * m;
    long long stride_b = (long long)k * n;
    long long stride_c = (long long)m * n;
    CUBLAS_CHECK(cublasSgemmStridedBatched(g_cublas_handle,
                                            CUBLAS_OP_N, CUBLAS_OP_T,
                                            n, m, k,
                                            &alpha, b, n, stride_b,
                                            a, m, stride_a,
                                            &beta, c, n, stride_c,
                                            batch_size));
    return 0;
}

extern "C" int fastnn_cuda_matmul_batched(const float* a, const float* b, float* c,
                                            int m, int n, int k, int batch_size,
                                            float alpha, float beta) {
    long long stride_a = (long long)m * k;
    long long stride_b = (long long)k * n;
    long long stride_c = (long long)m * n;

    CUBLAS_CHECK(cublasSgemmStridedBatched(g_cublas_handle,
                                            CUBLAS_OP_N, CUBLAS_OP_N,
                                            n, m, k,
                                            &alpha,
                                            b, n, stride_b,
                                            a, k, stride_a,
                                            &beta,
                                            c, n, stride_c,
                                            batch_size));
    return 0;
}

__global__ void kernel_transpose(const float* input, float* output, int rows, int cols) {
    __shared__ float tile[TILE_SIZE][TILE_SIZE + 1]; // +1 pad avoids bank conflicts

    int x = blockIdx.x * TILE_SIZE + threadIdx.x;
    int y = blockIdx.y * TILE_SIZE + threadIdx.y;

    if (x < cols && y < rows) {
        tile[threadIdx.y][threadIdx.x] = input[y * cols + x];
    }
    __syncthreads();

    x = blockIdx.y * TILE_SIZE + threadIdx.x;
    y = blockIdx.x * TILE_SIZE + threadIdx.y;

    if (x < rows && y < cols) {
        output[y * rows + x] = tile[threadIdx.x][threadIdx.y];
    }
}

extern "C" int fastnn_cuda_transpose(const float* input, float* output, int rows, int cols) {
    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks(div_ceil(cols, TILE_SIZE), div_ceil(rows, TILE_SIZE));
    kernel_transpose<<<blocks, threads>>>(input, output, rows, cols);
    CUDA_CHECK(cudaGetLastError());
    return 0;
}

__global__ void kernel_transpose_batched(const float* input, float* output, int batch_size, int rows, int cols) {
    int b = blockIdx.z;
    if (b >= batch_size) return;

    __shared__ float tile[TILE_SIZE][TILE_SIZE + 1];
    int offset = b * rows * cols;

    int x = blockIdx.x * TILE_SIZE + threadIdx.x;
    int y = blockIdx.y * TILE_SIZE + threadIdx.y;

    if (x < cols && y < rows)
        tile[threadIdx.y][threadIdx.x] = input[offset + y * cols + x];
    __syncthreads();

    x = blockIdx.y * TILE_SIZE + threadIdx.x;
    y = blockIdx.x * TILE_SIZE + threadIdx.y;

    if (x < rows && y < cols)
        output[offset + y * rows + x] = tile[threadIdx.x][threadIdx.y];
}

extern "C" int fastnn_cuda_transpose_batched(const float* input, float* output, int batch_size, int rows, int cols) {
    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks(div_ceil(cols, TILE_SIZE), div_ceil(rows, TILE_SIZE), batch_size);
    kernel_transpose_batched<<<blocks, threads>>>(input, output, batch_size, rows, cols);
    CUDA_CHECK(cudaGetLastError());
    return 0;
}

struct PermuteNDParams {
    int out_strides[PERMUTE_MAX_DIMS];
    int in_strides[PERMUTE_MAX_DIMS];
    int perm[PERMUTE_MAX_DIMS];
    int ndim;
};

__global__ void kernel_permute_nd(
    const float* __restrict__ input,
    float* __restrict__ output,
    PermuteNDParams params,
    int numel)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel) return;

    int rem = idx;
    int in_idx = 0;
    for (int d = 0; d < params.ndim; d++) {
        int coord = (params.out_strides[d] > 0) ? (rem / params.out_strides[d]) : 0;
        rem -= coord * params.out_strides[d];
        in_idx += coord * params.in_strides[params.perm[d]];
    }
    output[idx] = input[in_idx];
}

extern "C" int fastnn_cuda_permute_nd(
    const float* input, float* output,
    const int* out_strides, const int* in_strides, const int* perm,
    int ndim, int numel)
{
    PermuteNDParams params;
    params.ndim = ndim;
    for (int i = 0; i < ndim && i < PERMUTE_MAX_DIMS; i++) {
        params.out_strides[i] = out_strides[i];
        params.in_strides[i]  = in_strides[i];
        params.perm[i]        = perm[i];
    }
    int blocks = div_ceil(numel, BLOCK_SIZE);
    kernel_permute_nd<<<blocks, BLOCK_SIZE>>>(input, output, params, numel);
    CUDA_CHECK(cudaGetLastError());
    return 0;
}

__global__ void kernel_sum(const float* input, float* output, size_t n) {
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    float sum = 0.0f;
    if (i < n) sum += input[i];
    if (i + blockDim.x < n) sum += input[i + blockDim.x];
    sdata[tid] = sum;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > WARP_SIZE; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    if (tid < WARP_SIZE) {
        volatile float* smem = sdata;
        if (blockDim.x >= 64) smem[tid] += smem[tid + 32];
        smem[tid] += smem[tid + 16];
        smem[tid] += smem[tid + 8];
        smem[tid] += smem[tid + 4];
        smem[tid] += smem[tid + 2];
        smem[tid] += smem[tid + 1];
    }

    if (tid == 0) atomicAdd(output, sdata[0]);
}

extern "C" int fastnn_cuda_sum(const float* input, float* output, size_t n) {
    CUDA_CHECK(cudaMemset(output, 0, sizeof(float)));
    int blocks = div_ceil((int)n, BLOCK_SIZE * 2);
    kernel_sum<<<blocks, BLOCK_SIZE, BLOCK_SIZE * sizeof(float)>>>(input, output, n);
    CUDA_CHECK(cudaGetLastError());
    return 0;
}

__global__ void kernel_sum_axis(const float* input, float* output, int* shape, int ndim, int axis, int total_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int axis_size = shape[axis];
    int output_size = total_elements / axis_size;

    if (idx >= output_size) return;

    int inner_size = 1;
    for (int i = axis + 1; i < ndim; i++) inner_size *= shape[i];

    int outer_idx = idx / inner_size;
    int inner_idx = idx % inner_size;

    float sum = 0.0f;
    for (int a = 0; a < axis_size; a++) {
        int input_idx = (outer_idx * axis_size + a) * inner_size + inner_idx;
        sum += input[input_idx];
    }
    output[idx] = sum;
}

extern "C" int fastnn_cuda_sum_axis(const float* input, float* output, int* shape, int ndim, int axis, int total_elements) {
    // shape is host memory, the kernel needs its own device copy
    int* d_shape;
    CUDA_CHECK(cudaMalloc(&d_shape, ndim * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_shape, shape, ndim * sizeof(int), cudaMemcpyHostToDevice));

    int output_size = total_elements / shape[axis];

    int blocks = div_ceil(output_size, BLOCK_SIZE);
    kernel_sum_axis<<<blocks, BLOCK_SIZE>>>(input, output, d_shape, ndim, axis, total_elements);
    CUDA_CHECK(cudaGetLastError());
    cudaFree(d_shape);
    return 0;
}

extern "C" int fastnn_cuda_mean(const float* input, float* output, size_t n) {
    int ret = fastnn_cuda_sum(input, output, n);
    if (ret != 0) return ret;
    float inv_n = 1.0f / (float)n;
    kernel_mul_scalar<<<1, 1>>>(output, inv_n, output, 1);
    CUDA_CHECK(cudaGetLastError());
    return 0;
}

__global__ void kernel_max(const float* input, float* output, size_t n) {
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (i < n) ? input[i] : -FLT_MAX;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        __syncthreads();
    }

    if (tid == 0) {
        float old = *output;
        while (sdata[0] > old) {
            float assumed = old;
            old = __int_as_float(atomicCAS((int*)output, __float_as_int(assumed), __float_as_int(sdata[0])));
            if (old == assumed) break;
        }
    }
}

extern "C" int fastnn_cuda_max(const float* input, float* output, size_t n) {
    float neg_inf = -FLT_MAX;
    CUDA_CHECK(cudaMemcpy(output, &neg_inf, sizeof(float), cudaMemcpyHostToDevice));
    int blocks = div_ceil((int)n, BLOCK_SIZE);
    kernel_max<<<blocks, BLOCK_SIZE, BLOCK_SIZE * sizeof(float)>>>(input, output, n);
    CUDA_CHECK(cudaGetLastError());
    return 0;
}

extern "C" int fastnn_cuda_min(const float* input, float* output, size_t n) {
    float* temp;
    CUDA_CHECK(cudaMalloc(&temp, n * sizeof(float)));
    fastnn_cuda_neg(input, temp, n);
    fastnn_cuda_max(temp, output, n);
    fastnn_cuda_neg(output, output, 1);
    cudaFree(temp);
    return 0;
}

__global__ void kernel_im2col(
    const float* input, float* col,
    int channels, int height, int width,
    int kernel_h, int kernel_w,
    int stride_h, int stride_w,
    int pad_h, int pad_w,
    int out_h, int out_w)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = channels * kernel_h * kernel_w * out_h * out_w;
    if (idx >= total) return;

    int w_out = idx % out_w;
    int h_out = (idx / out_w) % out_h;
    int kw = (idx / out_w / out_h) % kernel_w;
    int kh = (idx / out_w / out_h / kernel_w) % kernel_h;
    int c = idx / out_w / out_h / kernel_w / kernel_h;

    int h_in = h_out * stride_h - pad_h + kh;
    int w_in = w_out * stride_w - pad_w + kw;

    col[idx] = (h_in >= 0 && h_in < height && w_in >= 0 && w_in < width)
               ? input[(c * height + h_in) * width + w_in]
               : 0.0f;
}

__global__ void kernel_col2im(
    const float* col, float* input,
    int channels, int height, int width,
    int kernel_h, int kernel_w,
    int stride_h, int stride_w,
    int pad_h, int pad_w,
    int out_h, int out_w)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = channels * height * width;
    if (idx >= total) return;

    int w = idx % width;
    int h = (idx / width) % height;
    int c = idx / width / height;

    float sum = 0.0f;
    for (int kh = 0; kh < kernel_h; kh++) {
        for (int kw = 0; kw < kernel_w; kw++) {
            int h_out_start = h + pad_h - kh;
            int w_out_start = w + pad_w - kw;
            if (h_out_start % stride_h == 0 && w_out_start % stride_w == 0) {
                int h_out = h_out_start / stride_h;
                int w_out = w_out_start / stride_w;
                if (h_out >= 0 && h_out < out_h && w_out >= 0 && w_out < out_w) {
                    int col_idx = ((c * kernel_h + kh) * kernel_w + kw) * out_h * out_w + h_out * out_w + w_out;
                    sum += col[col_idx];
                }
            }
        }
    }
    input[idx] = sum;
}

__global__ void kernel_layer_norm_forward(
    const float* input, const float* gamma, const float* beta,
    float* output, float* mean_out, float* inv_var_out,
    int batch_size, int normalized_size, float epsilon)
{
    int b = blockIdx.x;
    if (b >= batch_size) return;

    extern __shared__ float shared[];
    const float* in_row = input + b * normalized_size;
    float* out_row = output + b * normalized_size;

    float thread_sum = 0.0f;
    for (int i = threadIdx.x; i < normalized_size; i += blockDim.x)
        thread_sum += in_row[i];
    shared[threadIdx.x] = thread_sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) shared[threadIdx.x] += shared[threadIdx.x + s];
        __syncthreads();
    }
    float mean = shared[0] / (float)normalized_size;
    __syncthreads();

    float thread_var = 0.0f;
    for (int i = threadIdx.x; i < normalized_size; i += blockDim.x) {
        float diff = in_row[i] - mean;
        thread_var += diff * diff;
    }
    shared[threadIdx.x] = thread_var;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) shared[threadIdx.x] += shared[threadIdx.x + s];
        __syncthreads();
    }
    float inv_var = rsqrtf(shared[0] / (float)normalized_size + epsilon);

    if (threadIdx.x == 0) {
        mean_out[b] = mean;
        inv_var_out[b] = inv_var;
    }
    __syncthreads();

    for (int i = threadIdx.x; i < normalized_size; i += blockDim.x) {
        float normalized = (in_row[i] - mean) * inv_var;
        out_row[i] = gamma[i] * normalized + beta[i];
    }
}

extern "C" int fastnn_cuda_layer_norm_forward(
    const float* input, const float* gamma, const float* beta,
    float* output, float* mean, float* inv_var,
    int batch_size, int normalized_size, float epsilon)
{
    int threads = reduction_threads(normalized_size);
    kernel_layer_norm_forward<<<batch_size, threads, threads * sizeof(float)>>>(
        input, gamma, beta, output, mean, inv_var, batch_size, normalized_size, epsilon);
    CUDA_CHECK(cudaGetLastError());
    return 0;
}

__global__ void kernel_layer_norm_backward(
    const float* grad_output, const float* input,
    const float* gamma, const float* mean, const float* inv_var,
    float* grad_input, float* grad_gamma, float* grad_beta,
    int batch_size, int normalized_size)
{
    int b = blockIdx.x;
    if (b >= batch_size) return;

    extern __shared__ float shared[];
    const float* go_row = grad_output + b * normalized_size;
    const float* in_row = input + b * normalized_size;
    float* gi_row = grad_input + b * normalized_size;
    float m = mean[b];
    float iv = inv_var[b];

    float ds = 0.0f, db = 0.0f;
    for (int i = threadIdx.x; i < normalized_size; i += blockDim.x) {
        float xhat = (in_row[i] - m) * iv;
        ds += go_row[i] * gamma[i] * xhat;
        db += go_row[i] * gamma[i];
        atomicAdd(&grad_gamma[i], go_row[i] * xhat);
        atomicAdd(&grad_beta[i], go_row[i]);
    }

    float* shared2 = shared + blockDim.x;
    shared[threadIdx.x] = ds;
    shared2[threadIdx.x] = db;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            shared[threadIdx.x] += shared[threadIdx.x + s];
            shared2[threadIdx.x] += shared2[threadIdx.x + s];
        }
        __syncthreads();
    }
    ds = shared[0];
    db = shared2[0];
    __syncthreads();

    float inv_n = 1.0f / (float)normalized_size;
    for (int i = threadIdx.x; i < normalized_size; i += blockDim.x) {
        float xhat = (in_row[i] - m) * iv;
        gi_row[i] = iv * inv_n * ((float)normalized_size * go_row[i] * gamma[i] - db - xhat * ds);
    }
}

extern "C" int fastnn_cuda_layer_norm_backward(
    const float* grad_output, const float* input,
    const float* gamma, const float* mean, const float* inv_var,
    float* grad_input, float* grad_gamma, float* grad_beta,
    int batch_size, int normalized_size)
{
    CUDA_CHECK(cudaMemset(grad_gamma, 0, normalized_size * sizeof(float)));
    CUDA_CHECK(cudaMemset(grad_beta, 0, normalized_size * sizeof(float)));
    int threads = reduction_threads(normalized_size);
    kernel_layer_norm_backward<<<batch_size, threads, 2 * threads * sizeof(float)>>>(
        grad_output, input, gamma, mean, inv_var,
        grad_input, grad_gamma, grad_beta, batch_size, normalized_size);
    CUDA_CHECK(cudaGetLastError());
    return 0;
}

__global__ void kernel_embedding_forward(const int* indices, const float* weight, float* output, int num_indices, int embedding_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_indices * embedding_dim) return;

    int token_idx = idx / embedding_dim;
    int dim_idx = idx % embedding_dim;
    int vocab_idx = indices[token_idx];
    output[idx] = weight[vocab_idx * embedding_dim + dim_idx];
}

__global__ void kernel_embedding_backward(const int* indices, const float* grad_output, float* grad_weight,
                                           int num_indices, int embedding_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_indices * embedding_dim) return;

    int token_idx = idx / embedding_dim;
    int dim_idx = idx % embedding_dim;
    int vocab_idx = indices[token_idx];
    atomicAdd(&grad_weight[vocab_idx * embedding_dim + dim_idx], grad_output[idx]);
}

extern "C" int fastnn_cuda_embedding_forward(const int* indices, const float* weight, float* output, int num_indices, int embedding_dim) {
    int total = num_indices * embedding_dim;
    int blocks = div_ceil(total, BLOCK_SIZE);
    kernel_embedding_forward<<<blocks, BLOCK_SIZE>>>(indices, weight, output, num_indices, embedding_dim);
    CUDA_CHECK(cudaGetLastError());
    return 0;
}

extern "C" int fastnn_cuda_embedding_backward(const int* indices, const float* grad_output, float* grad_weight,
                                                int num_indices, int embedding_dim, int num_embeddings) {
    CUDA_CHECK(cudaMemset(grad_weight, 0, num_embeddings * embedding_dim * sizeof(float)));
    int total = num_indices * embedding_dim;
    int blocks = div_ceil(total, BLOCK_SIZE);
    kernel_embedding_backward<<<blocks, BLOCK_SIZE>>>(indices, grad_output, grad_weight, num_indices, embedding_dim);
    CUDA_CHECK(cudaGetLastError());
    return 0;
}

__global__ void kernel_im2col(const float* input, float* cols,
                              int c, int h, int w,
                              int kh, int kw, int sh, int sw, int ph, int pw,
                              int dh, int dw, int out_h, int out_w, int total) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;

    int positions = out_h * out_w;
    int patch = c * kh * kw;
    int pos = i % positions;
    int row = (i / positions) % patch;
    int image = i / (positions * patch);

    int ow = pos % out_w, oh = pos / out_w;
    int kj = row % kw, ki = (row / kw) % kh, channel = row / (kw * kh);

    int ih = oh * sh + ki * dh - ph;
    int iw = ow * sw + kj * dw - pw;

    float value = 0.0f;
    if (ih >= 0 && ih < h && iw >= 0 && iw < w)
        value = input[((image * c + channel) * h + ih) * w + iw];
    cols[i] = value;
}

extern "C" int fastnn_cuda_im2col(const float* input, float* cols,
                                  int n, int c, int h, int w,
                                  int kh, int kw, int sh, int sw, int ph, int pw,
                                  int dh, int dw, int out_h, int out_w) {
    int total = n * c * kh * kw * out_h * out_w;
    int blocks = div_ceil(total, BLOCK_SIZE);
    kernel_im2col<<<blocks, BLOCK_SIZE>>>(input, cols, c, h, w, kh, kw, sh, sw, ph, pw, dh, dw, out_h, out_w, total);
    CUDA_CHECK(cudaGetLastError());
    return 0;
}

// One thread per input pixel gathers its windows, so no atomics are needed.
__global__ void kernel_col2im(const float* cols, float* image,
                              int c, int h, int w,
                              int kh, int kw, int sh, int sw, int ph, int pw,
                              int dh, int dw, int out_h, int out_w, int total) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;

    int iw = i % w;
    int ih = (i / w) % h;
    int channel = (i / (w * h)) % c;
    int image_i = i / (w * h * c);

    int positions = out_h * out_w;
    int patch = c * kh * kw;

    float acc = 0.0f;
    for (int ki = 0; ki < kh; ki++) {
        int t = ih + ph - ki * dh;
        if (t < 0 || t % sh != 0) continue;
        int oh = t / sh;
        if (oh >= out_h) continue;
        for (int kj = 0; kj < kw; kj++) {
            int u = iw + pw - kj * dw;
            if (u < 0 || u % sw != 0) continue;
            int ow = u / sw;
            if (ow >= out_w) continue;
            int row = (channel * kh + ki) * kw + kj;
            acc += cols[(image_i * patch + row) * positions + oh * out_w + ow];
        }
    }
    image[i] = acc;
}

extern "C" int fastnn_cuda_col2im(const float* cols, float* image,
                                  int n, int c, int h, int w,
                                  int kh, int kw, int sh, int sw, int ph, int pw,
                                  int dh, int dw, int out_h, int out_w) {
    int total = n * c * h * w;
    int blocks = div_ceil(total, BLOCK_SIZE);
    kernel_col2im<<<blocks, BLOCK_SIZE>>>(cols, image, c, h, w, kh, kw, sh, sw, ph, pw, dh, dw, out_h, out_w, total);
    CUDA_CHECK(cudaGetLastError());
    return 0;
}
