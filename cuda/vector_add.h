#pragma once
#ifdef __cplusplus
extern "C"
{

#endif

void launch_vector_add(const float* a, const float* b, float* c, int n)
#ifdef __cplusplus
}
#endif

int cuda_alloc(void** ptr, size_t size);
int cuda_free(void* ptr);
int cuda_memcpy(void* dst, const void* src, size_t size, int kind);