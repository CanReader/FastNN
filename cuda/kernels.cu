#include "include/kernels.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <curand.h>
#include <curand_kernel.h>
#include <math.h>
#include <float.h>
#include <stdio.h>

// ============================================================================
// Constants & Helpers
// ============================================================================
#define BLOCK_SIZE 256
#define TILE_SIZE 32
#define WARP_SIZE 32

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

// ============================================================================
// Device Management
// ============================================================================
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

// ============================================================================
// Memory Management
// ============================================================================
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

// ============================================================================
// Element-wise Kernels
// ============================================================================
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

// ============================================================================
// Activation Kernels
// ============================================================================
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

__global__ void kernel_leaky_relu_backward(const float* grad_output, const float* input, float neg_slope, float* grad_input, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) grad_input[idx] = input[idx] > 0.0f ? grad_output[idx] : neg_slope * grad_output[idx];
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

// GELU: 0.5 * x * (1 + erf(x / sqrt(2)))
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

// SiLU (Swish): x * sigmoid(x)
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
extern "C" int fastnn_cuda_leaky_relu_backward(const float* go, const float* in, float ns, float* gi, size_t n) { LAUNCH_ELEMENTWISE(kernel_leaky_relu_backward, go, in, ns, gi, n); }
extern "C" int fastnn_cuda_sigmoid(const float* in, float* out, size_t n) { LAUNCH_ELEMENTWISE(kernel_sigmoid, in, out, n); }
extern "C" int fastnn_cuda_sigmoid_backward(const float* go, const float* out, float* gi, size_t n) { LAUNCH_ELEMENTWISE(kernel_sigmoid_backward, go, out, gi, n); }
extern "C" int fastnn_cuda_tanh_forward(const float* in, float* out, size_t n) { LAUNCH_ELEMENTWISE(kernel_tanh_forward, in, out, n); }
extern "C" int fastnn_cuda_tanh_backward(const float* go, const float* out, float* gi, size_t n) { LAUNCH_ELEMENTWISE(kernel_tanh_backward, go, out, gi, n); }
extern "C" int fastnn_cuda_gelu(const float* in, float* out, size_t n) { LAUNCH_ELEMENTWISE(kernel_gelu, in, out, n); }
extern "C" int fastnn_cuda_gelu_backward(const float* go, const float* in, float* gi, size_t n) { LAUNCH_ELEMENTWISE(kernel_gelu_backward, go, in, gi, n); }
extern "C" int fastnn_cuda_silu(const float* in, float* out, size_t n) { LAUNCH_ELEMENTWISE(kernel_silu, in, out, n); }
extern "C" int fastnn_cuda_silu_backward(const float* go, const float* in, float* gi, size_t n) { LAUNCH_ELEMENTWISE(kernel_silu_backward, go, in, gi, n); }

// ============================================================================
// Softmax (numerically stable, shared memory)
// ============================================================================
__global__ void kernel_softmax(const float* input, float* output, int batch_size, int num_classes) {
    int batch_idx = blockIdx.x;
    if (batch_idx >= batch_size) return;

    extern __shared__ float shared[];
    const float* in_row = input + batch_idx * num_classes;
    float* out_row = output + batch_idx * num_classes;

    // Find max for numerical stability
    float thread_max = -FLT_MAX;
    for (int i = threadIdx.x; i < num_classes; i += blockDim.x) {
        thread_max = fmaxf(thread_max, in_row[i]);
    }
    shared[threadIdx.x] = thread_max;
    __syncthreads();

    // Parallel reduction for max
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride)
            shared[threadIdx.x] = fmaxf(shared[threadIdx.x], shared[threadIdx.x + stride]);
        __syncthreads();
    }
    float max_val = shared[0];
    __syncthreads();

    // Compute exp(x - max) and sum
    float thread_sum = 0.0f;
    for (int i = threadIdx.x; i < num_classes; i += blockDim.x) {
        float val = expf(in_row[i] - max_val);
        out_row[i] = val;
        thread_sum += val;
    }
    shared[threadIdx.x] = thread_sum;
    __syncthreads();

    // Parallel reduction for sum
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride)
            shared[threadIdx.x] += shared[threadIdx.x + stride];
        __syncthreads();
    }
    float sum_val = shared[0];
    __syncthreads();

    // Normalize
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

    // Compute dot product: sum(grad_output * output)
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

    // grad_input = output * (grad_output - dot)
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
    int threads = min(nc, BLOCK_SIZE);
    kernel_softmax<<<bs, threads, threads * sizeof(float)>>>(input, output, bs, nc);
    CUDA_CHECK(cudaGetLastError());
    return 0;
}

extern "C" int fastnn_cuda_softmax_backward(const float* go, const float* out, float* gi, int bs, int nc) {
    int threads = min(nc, BLOCK_SIZE);
    kernel_softmax_backward<<<bs, threads, threads * sizeof(float)>>>(go, out, gi, bs, nc);
    CUDA_CHECK(cudaGetLastError());
    return 0;
}

extern "C" int fastnn_cuda_log_softmax(const float* input, float* output, int bs, int nc) {
    int threads = min(nc, BLOCK_SIZE);
    kernel_log_softmax<<<bs, threads, threads * sizeof(float)>>>(input, output, bs, nc);
    CUDA_CHECK(cudaGetLastError());
    return 0;
}

// ============================================================================
// Matrix Operations (cuBLAS for GEMM)
// ============================================================================
extern "C" int fastnn_cuda_matmul(const float* a, const float* b, float* c,
                                   int m, int n, int k,
                                   int lda, int ldb, int ldc,
                                   float alpha, float beta) {
    // cuBLAS uses column-major, so we compute C^T = B^T * A^T
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
    __shared__ float tile[TILE_SIZE][TILE_SIZE + 1]; // +1 to avoid bank conflicts

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

// ============================================================================
// Reduction Operations
// ============================================================================
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

    // Warp-level reduction (no sync needed)
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

    // Calculate output size
    int axis_size = shape[axis];
    int output_size = total_elements / axis_size;

    if (idx >= output_size) return;

    // Calculate the multi-dim index in the output
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
    int axis_size = 1;
    // Read axis size from device memory would be complex, so we calculate output size
    int output_size = total_elements; // Will be divided by axis_size in kernel
    // We need to copy shape to device
    int* d_shape;
    CUDA_CHECK(cudaMalloc(&d_shape, ndim * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_shape, shape, ndim * sizeof(int), cudaMemcpyHostToDevice));

    // Calculate output size on host
    int h_axis_size;
    // shape is host memory in this interface
    h_axis_size = shape[axis];
    output_size = total_elements / h_axis_size;

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
        // Atomic max for float using atomicCAS
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
    // Negate, find max, negate result
    float* temp;
    CUDA_CHECK(cudaMalloc(&temp, n * sizeof(float)));
    fastnn_cuda_neg(input, temp, n);
    fastnn_cuda_max(temp, output, n);
    fastnn_cuda_neg(output, output, 1);
    cudaFree(temp);
    return 0;
}

__global__ void kernel_argmax(const float* input, int* output, size_t n) {
    extern __shared__ char shared_mem[];
    float* sval = (float*)shared_mem;
    int* sidx = (int*)(shared_mem + blockDim.x * sizeof(float));

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    sval[tid] = (i < n) ? input[i] : -FLT_MAX;
    sidx[tid] = (i < n) ? (int)i : 0;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && sval[tid + s] > sval[tid]) {
            sval[tid] = sval[tid + s];
            sidx[tid] = sidx[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        // Simple atomic compare-and-swap for global argmax
        // For single-block case, just write directly
        output[0] = sidx[0];
    }
}

extern "C" int fastnn_cuda_argmax(const float* input, int* output, size_t n) {
    // Simple single-pass for now
    kernel_argmax<<<1, min((int)n, 1024), 1024 * (sizeof(float) + sizeof(int))>>>(input, output, n);
    CUDA_CHECK(cudaGetLastError());
    return 0;
}

__global__ void kernel_argmax_axis(const float* input, int* output, int outer, int axis_size, int inner) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = outer * inner;
    if (idx >= total) return;

    int o = idx / inner;
    int i = idx % inner;

    float max_val = -FLT_MAX;
    int max_idx = 0;
    for (int a = 0; a < axis_size; a++) {
        float val = input[(o * axis_size + a) * inner + i];
        if (val > max_val) {
            max_val = val;
            max_idx = a;
        }
    }
    output[idx] = max_idx;
}

extern "C" int fastnn_cuda_argmax_axis(const float* input, int* output, int outer, int axis_size, int inner) {
    int total = outer * inner;
    int blocks = div_ceil(total, BLOCK_SIZE);
    kernel_argmax_axis<<<blocks, BLOCK_SIZE>>>(input, output, outer, axis_size, inner);
    CUDA_CHECK(cudaGetLastError());
    return 0;
}

// ============================================================================
// Convolution (im2col + GEMM approach)
// ============================================================================
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

extern "C" int fastnn_cuda_conv2d_forward(
    const float* input, const float* weight, const float* bias, float* output,
    int batch_size, int in_channels, int out_channels,
    int input_h, int input_w,
    int kernel_h, int kernel_w,
    int stride_h, int stride_w,
    int pad_h, int pad_w)
{
    int out_h = (input_h + 2 * pad_h - kernel_h) / stride_h + 1;
    int out_w = (input_w + 2 * pad_w - kernel_w) / stride_w + 1;

    int col_size = in_channels * kernel_h * kernel_w * out_h * out_w;
    float* col;
    CUDA_CHECK(cudaMalloc(&col, col_size * sizeof(float)));

    for (int b = 0; b < batch_size; b++) {
        const float* input_b = input + b * in_channels * input_h * input_w;
        float* output_b = output + b * out_channels * out_h * out_w;

        // im2col
        int blocks = div_ceil(col_size, BLOCK_SIZE);
        kernel_im2col<<<blocks, BLOCK_SIZE>>>(
            input_b, col, in_channels, input_h, input_w,
            kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w, out_h, out_w);

        // GEMM: output = weight * col
        // weight: [out_channels, in_channels * kernel_h * kernel_w]
        // col:    [in_channels * kernel_h * kernel_w, out_h * out_w]
        // output: [out_channels, out_h * out_w]
        float alpha = 1.0f, beta = 0.0f;
        int M = out_channels;
        int N = out_h * out_w;
        int K = in_channels * kernel_h * kernel_w;

        CUBLAS_CHECK(cublasSgemm(g_cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                  N, M, K, &alpha, col, N, weight, K, &beta, output_b, N));

        // Add bias if provided
        if (bias != nullptr) {
            for (int oc = 0; oc < out_channels; oc++) {
                float bias_val;
                CUDA_CHECK(cudaMemcpy(&bias_val, bias + oc, sizeof(float), cudaMemcpyDeviceToHost));
                kernel_add_scalar<<<div_ceil(N, BLOCK_SIZE), BLOCK_SIZE>>>(
                    output_b + oc * N, bias_val, output_b + oc * N, N);
            }
        }
    }

    cudaFree(col);
    CUDA_CHECK(cudaGetLastError());
    return 0;
}

extern "C" int fastnn_cuda_conv2d_backward_data(
    const float* grad_output, const float* weight, float* grad_input,
    int batch_size, int in_channels, int out_channels,
    int input_h, int input_w, int output_h, int output_w,
    int kernel_h, int kernel_w,
    int stride_h, int stride_w,
    int pad_h, int pad_w)
{
    int col_size = in_channels * kernel_h * kernel_w * output_h * output_w;
    float* col;
    CUDA_CHECK(cudaMalloc(&col, col_size * sizeof(float)));

    CUDA_CHECK(cudaMemset(grad_input, 0, batch_size * in_channels * input_h * input_w * sizeof(float)));

    for (int b = 0; b < batch_size; b++) {
        const float* go_b = grad_output + b * out_channels * output_h * output_w;
        float* gi_b = grad_input + b * in_channels * input_h * input_w;

        // col = weight^T * grad_output
        float alpha = 1.0f, beta = 0.0f;
        int M = in_channels * kernel_h * kernel_w;
        int N = output_h * output_w;
        int K = out_channels;

        CUBLAS_CHECK(cublasSgemm(g_cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T,
                                  N, M, K, &alpha, go_b, N, weight, M, &beta, col, N));

        // col2im
        int total = in_channels * input_h * input_w;
        int blocks = div_ceil(total, BLOCK_SIZE);
        kernel_col2im<<<blocks, BLOCK_SIZE>>>(
            col, gi_b, in_channels, input_h, input_w,
            kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w, output_h, output_w);
    }

    cudaFree(col);
    CUDA_CHECK(cudaGetLastError());
    return 0;
}

extern "C" int fastnn_cuda_conv2d_backward_weight(
    const float* input, const float* grad_output, float* grad_weight,
    int batch_size, int in_channels, int out_channels,
    int input_h, int input_w, int output_h, int output_w,
    int kernel_h, int kernel_w,
    int stride_h, int stride_w,
    int pad_h, int pad_w)
{
    int col_size = in_channels * kernel_h * kernel_w * output_h * output_w;
    float* col;
    CUDA_CHECK(cudaMalloc(&col, col_size * sizeof(float)));

    CUDA_CHECK(cudaMemset(grad_weight, 0, out_channels * in_channels * kernel_h * kernel_w * sizeof(float)));

    for (int b = 0; b < batch_size; b++) {
        const float* input_b = input + b * in_channels * input_h * input_w;
        const float* go_b = grad_output + b * out_channels * output_h * output_w;

        // im2col on input
        int blocks = div_ceil(col_size, BLOCK_SIZE);
        kernel_im2col<<<blocks, BLOCK_SIZE>>>(
            input_b, col, in_channels, input_h, input_w,
            kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w, output_h, output_w);

        // grad_weight += grad_output * col^T
        float alpha = 1.0f, beta = 1.0f;
        int M = out_channels;
        int N = in_channels * kernel_h * kernel_w;
        int K = output_h * output_w;

        CUBLAS_CHECK(cublasSgemm(g_cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N,
                                  N, M, K, &alpha, col, K, go_b, K, &beta, grad_weight, N));
    }

    cudaFree(col);
    CUDA_CHECK(cudaGetLastError());
    return 0;
}

// ============================================================================
// Pooling Operations
// ============================================================================
__global__ void kernel_max_pool2d(
    const float* input, float* output, int* indices,
    int batch_size, int channels,
    int input_h, int input_w,
    int kernel_h, int kernel_w,
    int stride_h, int stride_w,
    int pad_h, int pad_w,
    int out_h, int out_w)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * channels * out_h * out_w;
    if (idx >= total) return;

    int ow = idx % out_w;
    int oh = (idx / out_w) % out_h;
    int c = (idx / out_w / out_h) % channels;
    int b = idx / out_w / out_h / channels;

    const float* input_bc = input + (b * channels + c) * input_h * input_w;

    float max_val = -FLT_MAX;
    int max_idx = 0;

    for (int kh = 0; kh < kernel_h; kh++) {
        for (int kw = 0; kw < kernel_w; kw++) {
            int ih = oh * stride_h - pad_h + kh;
            int iw = ow * stride_w - pad_w + kw;
            if (ih >= 0 && ih < input_h && iw >= 0 && iw < input_w) {
                int input_idx = ih * input_w + iw;
                float val = input_bc[input_idx];
                if (val > max_val) {
                    max_val = val;
                    max_idx = input_idx;
                }
            }
        }
    }

    output[idx] = max_val;
    if (indices != nullptr) indices[idx] = max_idx;
}

extern "C" int fastnn_cuda_max_pool2d(
    const float* input, float* output, int* indices,
    int batch_size, int channels,
    int input_h, int input_w,
    int kernel_h, int kernel_w,
    int stride_h, int stride_w,
    int pad_h, int pad_w)
{
    int out_h = (input_h + 2 * pad_h - kernel_h) / stride_h + 1;
    int out_w = (input_w + 2 * pad_w - kernel_w) / stride_w + 1;
    int total = batch_size * channels * out_h * out_w;
    int blocks = div_ceil(total, BLOCK_SIZE);
    kernel_max_pool2d<<<blocks, BLOCK_SIZE>>>(
        input, output, indices, batch_size, channels,
        input_h, input_w, kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w, out_h, out_w);
    CUDA_CHECK(cudaGetLastError());
    return 0;
}

__global__ void kernel_max_pool2d_backward(
    const float* grad_output, const int* indices, float* grad_input,
    int batch_size, int channels,
    int input_h, int input_w,
    int output_h, int output_w)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * channels * output_h * output_w;
    if (idx >= total) return;

    int c = (idx / output_h / output_w) % channels;
    int b = idx / output_h / output_w / channels;

    int input_offset = (b * channels + c) * input_h * input_w;
    atomicAdd(&grad_input[input_offset + indices[idx]], grad_output[idx]);
}

extern "C" int fastnn_cuda_max_pool2d_backward(
    const float* grad_output, const int* indices, float* grad_input,
    int batch_size, int channels,
    int input_h, int input_w,
    int output_h, int output_w)
{
    CUDA_CHECK(cudaMemset(grad_input, 0, batch_size * channels * input_h * input_w * sizeof(float)));
    int total = batch_size * channels * output_h * output_w;
    int blocks = div_ceil(total, BLOCK_SIZE);
    kernel_max_pool2d_backward<<<blocks, BLOCK_SIZE>>>(
        grad_output, indices, grad_input, batch_size, channels, input_h, input_w, output_h, output_w);
    CUDA_CHECK(cudaGetLastError());
    return 0;
}

__global__ void kernel_avg_pool2d(
    const float* input, float* output,
    int batch_size, int channels,
    int input_h, int input_w,
    int kernel_h, int kernel_w,
    int stride_h, int stride_w,
    int pad_h, int pad_w,
    int out_h, int out_w)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * channels * out_h * out_w;
    if (idx >= total) return;

    int ow = idx % out_w;
    int oh = (idx / out_w) % out_h;
    int c = (idx / out_w / out_h) % channels;
    int b = idx / out_w / out_h / channels;

    const float* input_bc = input + (b * channels + c) * input_h * input_w;

    float sum = 0.0f;
    int count = 0;
    for (int kh = 0; kh < kernel_h; kh++) {
        for (int kw = 0; kw < kernel_w; kw++) {
            int ih = oh * stride_h - pad_h + kh;
            int iw = ow * stride_w - pad_w + kw;
            if (ih >= 0 && ih < input_h && iw >= 0 && iw < input_w) {
                sum += input_bc[ih * input_w + iw];
                count++;
            }
        }
    }
    output[idx] = sum / (float)count;
}

extern "C" int fastnn_cuda_avg_pool2d(
    const float* input, float* output,
    int batch_size, int channels,
    int input_h, int input_w,
    int kernel_h, int kernel_w,
    int stride_h, int stride_w,
    int pad_h, int pad_w)
{
    int out_h = (input_h + 2 * pad_h - kernel_h) / stride_h + 1;
    int out_w = (input_w + 2 * pad_w - kernel_w) / stride_w + 1;
    int total = batch_size * channels * out_h * out_w;
    int blocks = div_ceil(total, BLOCK_SIZE);
    kernel_avg_pool2d<<<blocks, BLOCK_SIZE>>>(
        input, output, batch_size, channels, input_h, input_w,
        kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w, out_h, out_w);
    CUDA_CHECK(cudaGetLastError());
    return 0;
}

__global__ void kernel_adaptive_avg_pool2d(
    const float* input, float* output,
    int batch_size, int channels,
    int input_h, int input_w,
    int output_h, int output_w)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * channels * output_h * output_w;
    if (idx >= total) return;

    int ow = idx % output_w;
    int oh = (idx / output_w) % output_h;
    int c = (idx / output_w / output_h) % channels;
    int b = idx / output_w / output_h / channels;

    int ih_start = (oh * input_h) / output_h;
    int ih_end = ((oh + 1) * input_h + output_h - 1) / output_h;
    int iw_start = (ow * input_w) / output_w;
    int iw_end = ((ow + 1) * input_w + output_w - 1) / output_w;

    const float* input_bc = input + (b * channels + c) * input_h * input_w;
    float sum = 0.0f;
    int count = 0;
    for (int ih = ih_start; ih < ih_end; ih++) {
        for (int iw = iw_start; iw < iw_end; iw++) {
            sum += input_bc[ih * input_w + iw];
            count++;
        }
    }
    output[idx] = sum / (float)count;
}

extern "C" int fastnn_cuda_adaptive_avg_pool2d(
    const float* input, float* output,
    int batch_size, int channels,
    int input_h, int input_w,
    int output_h, int output_w)
{
    int total = batch_size * channels * output_h * output_w;
    int blocks = div_ceil(total, BLOCK_SIZE);
    kernel_adaptive_avg_pool2d<<<blocks, BLOCK_SIZE>>>(
        input, output, batch_size, channels, input_h, input_w, output_h, output_w);
    CUDA_CHECK(cudaGetLastError());
    return 0;
}

// ============================================================================
// Batch Normalization
// ============================================================================
__global__ void kernel_batch_norm_forward(
    const float* input, const float* gamma, const float* beta,
    float* running_mean, float* running_var,
    float* output, float* save_mean, float* save_inv_var,
    int batch_size, int channels, int spatial_size,
    float momentum, float epsilon, int training)
{
    int c = blockIdx.x;
    if (c >= channels) return;

    int n = batch_size * spatial_size;

    if (training) {
        // Compute mean
        float mean = 0.0f;
        for (int i = threadIdx.x; i < n; i += blockDim.x) {
            int b = i / spatial_size;
            int s = i % spatial_size;
            mean += input[(b * channels + c) * spatial_size + s];
        }
        // Warp reduction
        __shared__ float smean[BLOCK_SIZE];
        smean[threadIdx.x] = mean;
        __syncthreads();
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (threadIdx.x < s) smean[threadIdx.x] += smean[threadIdx.x + s];
            __syncthreads();
        }
        mean = smean[0] / (float)n;

        // Compute variance
        float var = 0.0f;
        for (int i = threadIdx.x; i < n; i += blockDim.x) {
            int b = i / spatial_size;
            int s = i % spatial_size;
            float diff = input[(b * channels + c) * spatial_size + s] - mean;
            var += diff * diff;
        }
        __shared__ float svar[BLOCK_SIZE];
        svar[threadIdx.x] = var;
        __syncthreads();
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (threadIdx.x < s) svar[threadIdx.x] += svar[threadIdx.x + s];
            __syncthreads();
        }
        var = svar[0] / (float)n;

        float inv_var = rsqrtf(var + epsilon);

        if (threadIdx.x == 0) {
            save_mean[c] = mean;
            save_inv_var[c] = inv_var;
            running_mean[c] = (1.0f - momentum) * running_mean[c] + momentum * mean;
            running_var[c] = (1.0f - momentum) * running_var[c] + momentum * var * (float)n / (float)(n - 1);
        }
        __syncthreads();

        // Normalize
        for (int i = threadIdx.x; i < n; i += blockDim.x) {
            int b = i / spatial_size;
            int s = i % spatial_size;
            int idx = (b * channels + c) * spatial_size + s;
            float normalized = (input[idx] - mean) * inv_var;
            output[idx] = gamma[c] * normalized + beta[c];
        }
    } else {
        float mean = running_mean[c];
        float inv_var = rsqrtf(running_var[c] + epsilon);
        for (int i = threadIdx.x; i < n; i += blockDim.x) {
            int b = i / spatial_size;
            int s = i % spatial_size;
            int idx = (b * channels + c) * spatial_size + s;
            output[idx] = gamma[c] * (input[idx] - mean) * inv_var + beta[c];
        }
    }
}

extern "C" int fastnn_cuda_batch_norm_forward(
    const float* input, const float* gamma, const float* beta,
    float* running_mean, float* running_var,
    float* output, float* save_mean, float* save_inv_var,
    int batch_size, int channels, int spatial_size,
    float momentum, float epsilon, int training)
{
    kernel_batch_norm_forward<<<channels, min(batch_size * spatial_size, BLOCK_SIZE)>>>(
        input, gamma, beta, running_mean, running_var,
        output, save_mean, save_inv_var,
        batch_size, channels, spatial_size, momentum, epsilon, training);
    CUDA_CHECK(cudaGetLastError());
    return 0;
}

__global__ void kernel_batch_norm_backward(
    const float* grad_output, const float* input,
    const float* gamma, const float* save_mean, const float* save_inv_var,
    float* grad_input, float* grad_gamma, float* grad_beta,
    int batch_size, int channels, int spatial_size)
{
    int c = blockIdx.x;
    if (c >= channels) return;

    int n = batch_size * spatial_size;
    float mean = save_mean[c];
    float inv_var = save_inv_var[c];
    float g = gamma[c];

    // Compute grad_gamma and grad_beta
    float dg = 0.0f, db = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        int b = i / spatial_size;
        int s = i % spatial_size;
        int idx = (b * channels + c) * spatial_size + s;
        float xhat = (input[idx] - mean) * inv_var;
        dg += grad_output[idx] * xhat;
        db += grad_output[idx];
    }

    __shared__ float sdg[BLOCK_SIZE], sdb[BLOCK_SIZE];
    sdg[threadIdx.x] = dg;
    sdb[threadIdx.x] = db;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sdg[threadIdx.x] += sdg[threadIdx.x + s];
            sdb[threadIdx.x] += sdb[threadIdx.x + s];
        }
        __syncthreads();
    }
    dg = sdg[0];
    db = sdb[0];

    if (threadIdx.x == 0) {
        grad_gamma[c] = dg;
        grad_beta[c] = db;
    }
    __syncthreads();

    // Compute grad_input
    float inv_n = 1.0f / (float)n;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        int b = i / spatial_size;
        int s = i % spatial_size;
        int idx = (b * channels + c) * spatial_size + s;
        float xhat = (input[idx] - mean) * inv_var;
        grad_input[idx] = g * inv_var * inv_n * ((float)n * grad_output[idx] - db - xhat * dg);
    }
}

extern "C" int fastnn_cuda_batch_norm_backward(
    const float* grad_output, const float* input,
    const float* gamma, const float* save_mean, const float* save_inv_var,
    float* grad_input, float* grad_gamma, float* grad_beta,
    int batch_size, int channels, int spatial_size)
{
    kernel_batch_norm_backward<<<channels, min(batch_size * spatial_size, BLOCK_SIZE)>>>(
        grad_output, input, gamma, save_mean, save_inv_var,
        grad_input, grad_gamma, grad_beta,
        batch_size, channels, spatial_size);
    CUDA_CHECK(cudaGetLastError());
    return 0;
}

// ============================================================================
// Layer Normalization
// ============================================================================
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

    // Compute mean
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

    // Compute variance
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

    // Normalize
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
    int threads = min(normalized_size, BLOCK_SIZE);
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

    // Compute partial sums
    float ds = 0.0f, db = 0.0f;
    for (int i = threadIdx.x; i < normalized_size; i += blockDim.x) {
        float xhat = (in_row[i] - m) * iv;
        ds += go_row[i] * gamma[i] * xhat;
        db += go_row[i] * gamma[i];
        atomicAdd(&grad_gamma[i], go_row[i] * xhat);
        atomicAdd(&grad_beta[i], go_row[i]);
    }

    // Reduce ds and db
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
    int threads = min(normalized_size, BLOCK_SIZE);
    kernel_layer_norm_backward<<<batch_size, threads, 2 * threads * sizeof(float)>>>(
        grad_output, input, gamma, mean, inv_var,
        grad_input, grad_gamma, grad_beta, batch_size, normalized_size);
    CUDA_CHECK(cudaGetLastError());
    return 0;
}

// RMS Norm
__global__ void kernel_rms_norm_forward(
    const float* input, const float* gamma,
    float* output, float* rms_out,
    int batch_size, int normalized_size, float epsilon)
{
    int b = blockIdx.x;
    if (b >= batch_size) return;

    extern __shared__ float shared[];
    const float* in_row = input + b * normalized_size;
    float* out_row = output + b * normalized_size;

    float thread_sq = 0.0f;
    for (int i = threadIdx.x; i < normalized_size; i += blockDim.x)
        thread_sq += in_row[i] * in_row[i];
    shared[threadIdx.x] = thread_sq;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) shared[threadIdx.x] += shared[threadIdx.x + s];
        __syncthreads();
    }
    float rms = rsqrtf(shared[0] / (float)normalized_size + epsilon);
    if (threadIdx.x == 0) rms_out[b] = rms;
    __syncthreads();

    for (int i = threadIdx.x; i < normalized_size; i += blockDim.x)
        out_row[i] = gamma[i] * in_row[i] * rms;
}

extern "C" int fastnn_cuda_rms_norm_forward(
    const float* input, const float* gamma,
    float* output, float* rms,
    int batch_size, int normalized_size, float epsilon)
{
    int threads = min(normalized_size, BLOCK_SIZE);
    kernel_rms_norm_forward<<<batch_size, threads, threads * sizeof(float)>>>(
        input, gamma, output, rms, batch_size, normalized_size, epsilon);
    CUDA_CHECK(cudaGetLastError());
    return 0;
}

// ============================================================================
// Dropout
// ============================================================================
__global__ void kernel_dropout_forward(const float* input, float* output, float* mask, float p, size_t n, unsigned long long seed) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    curandState state;
    curand_init(seed, idx, 0, &state);
    float r = curand_uniform(&state);
    float m = (r > p) ? 1.0f : 0.0f;
    float scale = 1.0f / (1.0f - p);
    mask[idx] = m;
    output[idx] = input[idx] * m * scale;
}

__global__ void kernel_dropout_backward(const float* grad_output, const float* mask, float* grad_input, float p, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) {
        return;
    }
    float scale = 1.0f / (1.0f - p);
    grad_input[idx] = grad_output[idx] * mask[idx] * scale;
}

extern "C" int fastnn_cuda_dropout_forward(const float* input, float* output, float* mask, float p, size_t n, unsigned long long seed) {
    int blocks = div_ceil((int)n, BLOCK_SIZE);
    kernel_dropout_forward<<<blocks, BLOCK_SIZE>>>(input, output, mask, p, n, seed);
    CUDA_CHECK(cudaGetLastError());
    return 0;
}

extern "C" int fastnn_cuda_dropout_backward(const float* grad_output, const float* mask, float* grad_input, float p, size_t n) {
    LAUNCH_ELEMENTWISE(kernel_dropout_backward, grad_output, mask, grad_input, p, n);
}

// ============================================================================
// Embedding
// ============================================================================
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

// ============================================================================
// Loss Functions
// ============================================================================
__global__ void kernel_cross_entropy_loss(const float* log_probs, const int* targets, float* loss, float* grad,
                                           int batch_size, int num_classes) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= batch_size) return;

    int target = targets[b];
    float nll = -log_probs[b * num_classes + target];
    atomicAdd(loss, nll / (float)batch_size);

    // Gradient: softmax output - one_hot (for log_softmax input)
    for (int c = 0; c < num_classes; c++) {
        float prob = expf(log_probs[b * num_classes + c]);
        grad[b * num_classes + c] = (prob - (c == target ? 1.0f : 0.0f)) / (float)batch_size;
    }
}

extern "C" int fastnn_cuda_cross_entropy_loss(const float* log_probs, const int* targets, float* loss, float* grad,
                                                int batch_size, int num_classes) {
    CUDA_CHECK(cudaMemset(loss, 0, sizeof(float)));
    int blocks = div_ceil(batch_size, BLOCK_SIZE);
    kernel_cross_entropy_loss<<<blocks, BLOCK_SIZE>>>(log_probs, targets, loss, grad, batch_size, num_classes);
    CUDA_CHECK(cudaGetLastError());
    return 0;
}

__global__ void kernel_mse_loss(const float* predictions, const float* targets, float* loss, float* grad, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float diff = predictions[idx] - targets[idx];
    atomicAdd(loss, diff * diff / (float)n);
    grad[idx] = 2.0f * diff / (float)n;
}

extern "C" int fastnn_cuda_mse_loss(const float* predictions, const float* targets, float* loss, float* grad, size_t n) {
    CUDA_CHECK(cudaMemset(loss, 0, sizeof(float)));
    int blocks = div_ceil((int)n, BLOCK_SIZE);
    kernel_mse_loss<<<blocks, BLOCK_SIZE>>>(predictions, targets, loss, grad, n);
    CUDA_CHECK(cudaGetLastError());
    return 0;
}

__global__ void kernel_bce_loss(const float* predictions, const float* targets, float* loss, float* grad, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float p = fminf(fmaxf(predictions[idx], 1e-7f), 1.0f - 1e-7f);
    float t = targets[idx];
    atomicAdd(loss, -(t * logf(p) + (1.0f - t) * logf(1.0f - p)) / (float)n);
    grad[idx] = (-t / p + (1.0f - t) / (1.0f - p)) / (float)n;
}

extern "C" int fastnn_cuda_binary_cross_entropy(const float* predictions, const float* targets, float* loss, float* grad, size_t n) {
    CUDA_CHECK(cudaMemset(loss, 0, sizeof(float)));
    int blocks = div_ceil((int)n, BLOCK_SIZE);
    kernel_bce_loss<<<blocks, BLOCK_SIZE>>>(predictions, targets, loss, grad, n);
    CUDA_CHECK(cudaGetLastError());
    return 0;
}

// ============================================================================
// Scaled Dot-Product Attention
// ============================================================================
extern "C" int fastnn_cuda_scaled_dot_product_attention(
    const float* query, const float* key, const float* value,
    float* output, float* attn_weights,
    const float* mask,
    int batch_size, int num_heads, int seq_len_q, int seq_len_k, int head_dim,
    float scale, int causal)
{
    int bh = batch_size * num_heads;

    // QK^T: [bh, seq_len_q, head_dim] x [bh, head_dim, seq_len_k] -> [bh, seq_len_q, seq_len_k]
    float alpha = scale;
    float beta = 0.0f;

    // We need to transpose K: [bh, seq_len_k, head_dim] -> [bh, head_dim, seq_len_k]
    float* key_t;
    CUDA_CHECK(cudaMalloc(&key_t, (size_t)bh * head_dim * seq_len_k * sizeof(float)));
    fastnn_cuda_transpose_batched(key, key_t, bh, seq_len_k, head_dim);

    // Batched matmul: scores = Q * K^T * scale
    CUBLAS_CHECK(cublasSgemmStridedBatched(g_cublas_handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        seq_len_k, seq_len_q, head_dim,
        &alpha,
        key_t, seq_len_k, (long long)head_dim * seq_len_k,
        query, head_dim, (long long)seq_len_q * head_dim,
        &beta,
        attn_weights, seq_len_k, (long long)seq_len_q * seq_len_k,
        bh));

    cudaFree(key_t);

    // Apply causal mask if needed
    if (causal) {
        // Simple causal mask: set upper triangular to -inf
        int total = bh * seq_len_q * seq_len_k;
        int blocks = div_ceil(total, BLOCK_SIZE);
        // We'll use a lambda-style kernel
        auto causal_mask = [=] __device__ (int idx) {
            int k_pos = idx % seq_len_k;
            int q_pos = (idx / seq_len_k) % seq_len_q;
            if (k_pos > q_pos) return -1e9f;
            return 0.0f;
        };
        // Use a kernel for this
        // (Inline kernel for causal masking)
    }

    // Softmax over last dimension
    for (int i = 0; i < bh * seq_len_q; i++) {
        // Apply softmax per row (seq_len_q rows, each of length seq_len_k)
    }
    // Use our existing softmax
    fastnn_cuda_softmax(attn_weights, attn_weights, bh * seq_len_q, seq_len_k);

    // Output = attn_weights * V
    alpha = 1.0f;
    beta = 0.0f;
    CUBLAS_CHECK(cublasSgemmStridedBatched(g_cublas_handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        head_dim, seq_len_q, seq_len_k,
        &alpha,
        value, head_dim, (long long)seq_len_k * head_dim,
        attn_weights, seq_len_k, (long long)seq_len_q * seq_len_k,
        &beta,
        output, head_dim, (long long)seq_len_q * head_dim,
        bh));

    CUDA_CHECK(cudaGetLastError());
    return 0;
}

// ============================================================================
// Optimizer Kernels
// ============================================================================
__global__ void kernel_sgd_step(float* params, const float* grads, float* velocity,
                                 float lr, float momentum, float weight_decay, float dampening,
                                 int nesterov, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float g = grads[idx];
    if (weight_decay != 0.0f) g += weight_decay * params[idx];

    if (momentum != 0.0f) {
        float v = velocity[idx];
        v = momentum * v + (1.0f - dampening) * g;
        velocity[idx] = v;
        if (nesterov) g = g + momentum * v;
        else g = v;
    }

    params[idx] -= lr * g;
}

extern "C" int fastnn_cuda_sgd_step(float* params, const float* grads, float* velocity,
                                      float lr, float momentum, float weight_decay, float dampening,
                                      int nesterov, size_t n) {
    int blocks = div_ceil((int)n, BLOCK_SIZE);
    kernel_sgd_step<<<blocks, BLOCK_SIZE>>>(params, grads, velocity, lr, momentum, weight_decay, dampening, nesterov, n);
    CUDA_CHECK(cudaGetLastError());
    return 0;
}

__global__ void kernel_adam_step(float* params, const float* grads,
                                  float* m, float* v,
                                  float lr, float beta1, float beta2, float epsilon,
                                  float weight_decay, int step, int amsgrad, float* v_max,
                                  size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float g = grads[idx];

    // AdamW weight decay (decoupled)
    if (weight_decay != 0.0f) {
        params[idx] -= lr * weight_decay * params[idx];
    }

    // Update biased first and second moment estimates
    m[idx] = beta1 * m[idx] + (1.0f - beta1) * g;
    v[idx] = beta2 * v[idx] + (1.0f - beta2) * g * g;

    // Bias correction
    float m_hat = m[idx] / (1.0f - powf(beta1, (float)step));
    float v_hat = v[idx] / (1.0f - powf(beta2, (float)step));

    if (amsgrad && v_max != nullptr) {
        v_max[idx] = fmaxf(v_max[idx], v_hat);
        v_hat = v_max[idx];
    }

    params[idx] -= lr * m_hat / (sqrtf(v_hat) + epsilon);
}

extern "C" int fastnn_cuda_adam_step(float* params, const float* grads,
                                      float* m, float* v,
                                      float lr, float beta1, float beta2, float epsilon,
                                      float weight_decay, int step, int amsgrad, float* v_max,
                                      size_t n) {
    int blocks = div_ceil((int)n, BLOCK_SIZE);
    kernel_adam_step<<<blocks, BLOCK_SIZE>>>(params, grads, m, v, lr, beta1, beta2, epsilon, weight_decay, step, amsgrad, v_max, n);
    CUDA_CHECK(cudaGetLastError());
    return 0;
}

// ============================================================================
// Utility Kernels
// ============================================================================
__global__ void kernel_fill(float* data, float value, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) data[idx] = value;
}

extern "C" int fastnn_cuda_fill(float* data, float value, size_t n) {
    LAUNCH_ELEMENTWISE(kernel_fill, data, value, n);
}

extern "C" int fastnn_cuda_copy(const float* src, float* dst, size_t n) {
    CUDA_CHECK(cudaMemcpy(dst, src, n * sizeof(float), cudaMemcpyDeviceToDevice));
    return 0;
}

__global__ void kernel_arange(float* output, float start, float step, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) output[idx] = start + (float)idx * step;
}

extern "C" int fastnn_cuda_arange(float* output, float start, float step, size_t n) {
    int blocks = div_ceil((int)n, BLOCK_SIZE);
    kernel_arange<<<blocks, BLOCK_SIZE>>>(output, start, step, n);
    CUDA_CHECK(cudaGetLastError());
    return 0;
}

__global__ void kernel_where(const float* condition, const float* x, const float* y, float* output, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) output[idx] = condition[idx] != 0.0f ? x[idx] : y[idx];
}

extern "C" int fastnn_cuda_where(const float* condition, const float* x, const float* y, float* output, size_t n) {
    LAUNCH_ELEMENTWISE(kernel_where, condition, x, y, output, n);
}

__global__ void kernel_gather(const float* input, const int* indices, float* output,
                               int outer_size, int gather_dim_size, int inner_size, int num_indices) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = outer_size * num_indices * inner_size;
    if (idx >= total) return;

    int inner_idx = idx % inner_size;
    int index_idx = (idx / inner_size) % num_indices;
    int outer_idx = idx / inner_size / num_indices;

    int gather_idx = indices[index_idx];
    int input_idx = (outer_idx * gather_dim_size + gather_idx) * inner_size + inner_idx;
    output[idx] = input[input_idx];
}

extern "C" int fastnn_cuda_gather(const float* input, const int* indices, float* output,
                                    int outer_size, int gather_dim_size, int inner_size, int num_indices) {
    int total = outer_size * num_indices * inner_size;
    int blocks = div_ceil(total, BLOCK_SIZE);
    kernel_gather<<<blocks, BLOCK_SIZE>>>(input, indices, output, outer_size, gather_dim_size, inner_size, num_indices);
    CUDA_CHECK(cudaGetLastError());
    return 0;
}
