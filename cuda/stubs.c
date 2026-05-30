/* CPU-only build stubs for CUDA kernel symbols.
 * Compiled when the `cuda` feature is disabled. These functions are never
 * called at runtime (all tensor storage is CPU in that configuration), but
 * the Rust FFI declarations require symbols to be present at link time.
 */
#include <stddef.h>

#define STUB return -1

int fastnn_cuda_malloc(float **ptr, size_t n)                   { (void)ptr;(void)n; STUB; }
int fastnn_cuda_free(float *ptr)                                 { (void)ptr; STUB; }
int fastnn_cuda_memcpy_h2d(float *d, const float *h, size_t n)  { (void)d;(void)h;(void)n; STUB; }
int fastnn_cuda_memcpy_d2h(float *h, const float *d, size_t n)  { (void)h;(void)d;(void)n; STUB; }
int fastnn_cuda_memcpy_d2d(float *d, const float *s, size_t n)  { (void)d;(void)s;(void)n; STUB; }
int fastnn_cuda_memset(float *p, int v, size_t n)               { (void)p;(void)v;(void)n; STUB; }
int fastnn_cuda_fill(float *p, float v, size_t n)               { (void)p;(void)v;(void)n; STUB; }

int fastnn_cuda_add(const float*a,const float*b,float*o,size_t n)            {(void)a;(void)b;(void)o;(void)n;STUB;}
int fastnn_cuda_sub(const float*a,const float*b,float*o,size_t n)            {(void)a;(void)b;(void)o;(void)n;STUB;}
int fastnn_cuda_mul(const float*a,const float*b,float*o,size_t n)            {(void)a;(void)b;(void)o;(void)n;STUB;}
int fastnn_cuda_div(const float*a,const float*b,float*o,size_t n)            {(void)a;(void)b;(void)o;(void)n;STUB;}
int fastnn_cuda_add_scalar(const float*a,float s,float*o,size_t n)           {(void)a;(void)s;(void)o;(void)n;STUB;}
int fastnn_cuda_mul_scalar(const float*a,float s,float*o,size_t n)           {(void)a;(void)s;(void)o;(void)n;STUB;}
int fastnn_cuda_pow_scalar(const float*a,float s,float*o,size_t n)           {(void)a;(void)s;(void)o;(void)n;STUB;}
int fastnn_cuda_sqrt(const float*a,float*o,size_t n)                         {(void)a;(void)o;(void)n;STUB;}
int fastnn_cuda_abs(const float*a,float*o,size_t n)                          {(void)a;(void)o;(void)n;STUB;}
int fastnn_cuda_neg(const float*a,float*o,size_t n)                          {(void)a;(void)o;(void)n;STUB;}
int fastnn_cuda_exp(const float*a,float*o,size_t n)                          {(void)a;(void)o;(void)n;STUB;}
int fastnn_cuda_log(const float*a,float*o,size_t n)                          {(void)a;(void)o;(void)n;STUB;}
int fastnn_cuda_clamp(const float*a,float lo,float hi,float*o,size_t n)      {(void)a;(void)lo;(void)hi;(void)o;(void)n;STUB;}
int fastnn_cuda_relu(const float*i,float*o,size_t n)                         {(void)i;(void)o;(void)n;STUB;}
int fastnn_cuda_relu_backward(const float*g,const float*i,float*gi,size_t n) {(void)g;(void)i;(void)gi;(void)n;STUB;}
int fastnn_cuda_sigmoid(const float*i,float*o,size_t n)                      {(void)i;(void)o;(void)n;STUB;}
int fastnn_cuda_sigmoid_backward(const float*g,const float*out,float*gi,size_t n){(void)g;(void)out;(void)gi;(void)n;STUB;}
int fastnn_cuda_tanh_forward(const float*i,float*o,size_t n)                 {(void)i;(void)o;(void)n;STUB;}
int fastnn_cuda_tanh_backward(const float*g,const float*out,float*gi,size_t n){(void)g;(void)out;(void)gi;(void)n;STUB;}
int fastnn_cuda_gelu(const float*i,float*o,size_t n)                         {(void)i;(void)o;(void)n;STUB;}
int fastnn_cuda_gelu_backward(const float*g,const float*i,float*gi,size_t n) {(void)g;(void)i;(void)gi;(void)n;STUB;}
int fastnn_cuda_silu(const float*i,float*o,size_t n)                         {(void)i;(void)o;(void)n;STUB;}
int fastnn_cuda_silu_backward(const float*g,const float*i,float*gi,size_t n) {(void)g;(void)i;(void)gi;(void)n;STUB;}
int fastnn_cuda_leaky_relu(const float*i,float neg,float*o,size_t n)         {(void)i;(void)neg;(void)o;(void)n;STUB;}
int fastnn_cuda_softmax(const float*i,float*o,size_t rows,size_t cols)       {(void)i;(void)o;(void)rows;(void)cols;STUB;}
int fastnn_cuda_softmax_backward(const float*g,const float*o,float*gi,int bs,int nc){(void)g;(void)o;(void)gi;(void)bs;(void)nc;STUB;}
int fastnn_cuda_log_softmax(const float*i,float*o,size_t rows,size_t cols)   {(void)i;(void)o;(void)rows;(void)cols;STUB;}
int fastnn_cuda_layer_norm_forward(const float*i,const float*g,const float*b,float*o,float*m,float*iv,int bs,int ns,float eps){(void)i;(void)g;(void)b;(void)o;(void)m;(void)iv;(void)bs;(void)ns;(void)eps;STUB;}
int fastnn_cuda_layer_norm_backward(const float*go,const float*i,const float*g,const float*m,const float*iv,float*gi,float*gg,float*gb,int bs,int ns){(void)go;(void)i;(void)g;(void)m;(void)iv;(void)gi;(void)gg;(void)gb;(void)bs;(void)ns;STUB;}
int fastnn_cuda_embedding_forward(const int*idx,const float*w,float*o,int n,int d){(void)idx;(void)w;(void)o;(void)n;(void)d;STUB;}
int fastnn_cuda_embedding_backward(const int*idx,const float*g,float*gw,int n,int d,int v){(void)idx;(void)g;(void)gw;(void)n;(void)d;(void)v;STUB;}
int fastnn_cuda_sum_axis(const float*i,float*o,const int*s,int nd,int ax,int n){(void)i;(void)o;(void)s;(void)nd;(void)ax;(void)n;STUB;}
int fastnn_cuda_sum(const float*i,float*o,size_t n)                          {(void)i;(void)o;(void)n;STUB;}
int fastnn_cuda_mean(const float*i,float*o,size_t n)                         {(void)i;(void)o;(void)n;STUB;}
int fastnn_cuda_max(const float*i,float*o,size_t n)                          {(void)i;(void)o;(void)n;STUB;}
int fastnn_cuda_min(const float*i,float*o,size_t n)                          {(void)i;(void)o;(void)n;STUB;}
int fastnn_cuda_argmax(const float*i,int*o,size_t n)                         {(void)i;(void)o;(void)n;STUB;}
int fastnn_cuda_argmax_axis(const float*i,int*o,size_t rows,size_t cols,int axis){(void)i;(void)o;(void)rows;(void)cols;(void)axis;STUB;}
int fastnn_cuda_matmul(const float*a,const float*b,float*c,int m,int k,int n){(void)a;(void)b;(void)c;(void)m;(void)k;(void)n;STUB;}
int fastnn_cuda_matmul_batched(const float*a,const float*b,float*c,int batch,int m,int k,int n){(void)a;(void)b;(void)c;(void)batch;(void)m;(void)k;(void)n;STUB;}
int fastnn_cuda_matmul_nt(const float*a,const float*b,float*c,int m,int n,int k){(void)a;(void)b;(void)c;(void)m;(void)n;(void)k;STUB;}
int fastnn_cuda_matmul_tn(const float*a,const float*b,float*c,int m,int n,int k){(void)a;(void)b;(void)c;(void)m;(void)n;(void)k;STUB;}
int fastnn_cuda_matmul_batched_nt(const float*a,const float*b,float*c,int m,int n,int k,int batch){(void)a;(void)b;(void)c;(void)m;(void)n;(void)k;(void)batch;STUB;}
int fastnn_cuda_matmul_batched_tn(const float*a,const float*b,float*c,int m,int n,int k,int batch){(void)a;(void)b;(void)c;(void)m;(void)n;(void)k;(void)batch;STUB;}
int fastnn_cuda_transpose(const float*i,float*o,int rows,int cols)           {(void)i;(void)o;(void)rows;(void)cols;STUB;}
int fastnn_cuda_transpose_batched(const float*i,float*o,int b,int rows,int cols){(void)i;(void)o;(void)b;(void)rows;(void)cols;STUB;}
int fastnn_cuda_permute_nd(const float*i,float*o,const int*os,const int*is,const int*p,int nd,int n){(void)i;(void)o;(void)os;(void)is;(void)p;(void)nd;(void)n;STUB;}
int fastnn_cuda_init(int device_id)                                           {(void)device_id;STUB;}
int fastnn_cuda_device_count(void)                                            {return 0;}
int fastnn_cuda_synchronize(void)                                             {STUB;}
void fastnn_cuda_get_memory_info(size_t*free,size_t*total)                   {if(free)*free=0;if(total)*total=0;}
