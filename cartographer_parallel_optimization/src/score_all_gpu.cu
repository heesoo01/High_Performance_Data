#include "cartographer_parallel/assignment.h"
#include <cuda_runtime.h>
#include <vector>
#include <iostream>
#include <chrono>

namespace cartographer_parallel {

// ═══════════════════════════════════════════════════════════
// GPU shmem 커널
// - thread 1개 = candidate 1개
// - block 내 모든 thread가 px/py를 shared memory에 협력 적재
//   → global memory 접근을 N×P회에서 block당 P회로 감소
// ═══════════════════════════════════════════════════════════
__global__ void score_all_kernel_shmem(
    const unsigned char* __restrict__ grid,
    int w, int h,
    const int* __restrict__ px,
    const int* __restrict__ py,
    const int* __restrict__ cx,
    const int* __restrict__ cy,
    float* __restrict__ score,
    int n, int p) {

  extern __shared__ int s[];
  int* s_px = s;
  int* s_py = s + p;

  // 협력 적재: block 내 thread들이 px/py를 나눠서 shared memory에 올림
  for (int j = threadIdx.x; j < p; j += blockDim.x) {
    s_px[j] = px[j];
    s_py[j] = py[j];
  }
  __syncthreads();  // 모든 thread의 로드 완료 대기

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;

  int sum = 0;
  const int cxi = cx[i];
  const int cyi = cy[i];
  const unsigned int uw = static_cast<unsigned int>(w);
  const unsigned int uh = static_cast<unsigned int>(h);

  for (int j = 0; j < p; ++j) {
    const unsigned int x = static_cast<unsigned int>(s_px[j] + cxi);
    const unsigned int y = static_cast<unsigned int>(s_py[j] + cyi);
    if (x < uw && y < uh)
      sum += grid[y * w + x];
  }
  score[i] = static_cast<float>(sum) / (255.0f * static_cast<float>(p));
}

// ═══════════════════════════════════════════════════════════
// GPU shmem wrapper (cudaMalloc 매 호출마다)
// ═══════════════════════════════════════════════════════════
void score_all_gpu(const std::vector<unsigned char>& grid, const int w,
                   const int h, const std::vector<int>& px,
                   const std::vector<int>& py, const std::vector<int>& cx,
                   const std::vector<int>& cy,
                   std::vector<float>* const score) {
  if (!score) return;
  const int n = static_cast<int>(cx.size());
  const int p = static_cast<int>(px.size());
  score->assign(n, 0.0f);
  if (w <= 0 || h <= 0 || p == 0 ||
      grid.size() < static_cast<size_t>(w * h)) return;

  unsigned char* d_grid; int* d_px; int* d_py;
  int* d_cx; int* d_cy; float* d_score;

  cudaMalloc(&d_grid,  grid.size() * sizeof(unsigned char));
  cudaMalloc(&d_px,    p * sizeof(int));
  cudaMalloc(&d_py,    p * sizeof(int));
  cudaMalloc(&d_cx,    n * sizeof(int));
  cudaMalloc(&d_cy,    n * sizeof(int));
  cudaMalloc(&d_score, n * sizeof(float));

  cudaMemcpy(d_grid, grid.data(), grid.size()*sizeof(unsigned char), cudaMemcpyHostToDevice);
  cudaMemcpy(d_px,   px.data(),   p*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_py,   py.data(),   p*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_cx,   cx.data(),   n*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_cy,   cy.data(),   n*sizeof(int), cudaMemcpyHostToDevice);

  const int threads = 256;
  const int blocks  = (n + threads - 1) / threads;
  const size_t shared_mem = 2 * p * sizeof(int);

  auto t0 = std::chrono::high_resolution_clock::now();
  score_all_kernel_shmem<<<blocks, threads, shared_mem>>>(
      d_grid, w, h, d_px, d_py, d_cx, d_cy, d_score, n, p);
  cudaDeviceSynchronize();
  cudaMemcpy(score->data(), d_score, n*sizeof(float), cudaMemcpyDeviceToHost);
  auto t1 = std::chrono::high_resolution_clock::now();

  auto us = std::chrono::duration_cast<std::chrono::microseconds>(t1-t0).count();
  std::cout << "[SCORE_ALL_gpu_shmem] candidates=" << n
            << ", scan_points=" << p
            << ", time=" << us << " us\n";

  cudaFree(d_grid); cudaFree(d_px); cudaFree(d_py);
  cudaFree(d_cx);   cudaFree(d_cy); cudaFree(d_score);
}

}  // namespace cartographer_parallel
