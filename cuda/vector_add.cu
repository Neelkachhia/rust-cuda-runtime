#include<cuda_runtime.h>
#include<stdio.h>

__global__ void vector_add(const float* a, const float* b, float* c, int n)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx < n)
    c[idx] = a[idx] + b[idx];
}

extern "C" void launch_vector_add(const float* a, const float* b, float* c, int n)
{
  int threads = 256;
  int blocks = (n + threads - 1) / threads;

  vector_add<<<blocks, threads>>>(a,b,c,n);
  cudaDeviceSynchronize();
}

extern "C" int cuda_alloc(void** ptr, size_t size)
{
  return cudaMalloc(ptr, size);
}

extern "C" int cuda_free(void* ptr)
{
  return cudaFree(ptr);
}

extern "C" int cuda_memcpy(void* dst, const void* src, size_t size, int kind)
{
  return cudaMemcpy(dst, src, size, static_cast<cudaMemcpyKind>(kind));
}