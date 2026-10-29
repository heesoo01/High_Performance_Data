#include "cartographer_parallel/assignment.h"
#include <cuda_runtime.h>
#include <vector>
#include <iostream>
#include <chrono>

namespace cartographer_parallel {

// ═══════════════════════════════════════════════════════════
// GPU v4 커널: shmem + __ldg + #pragma unroll
//
// __ldg: "이 데이터는 안 바뀐다"고 GPU에 알려줘서 읽기 전용 캐시 활용
//         Maxwell GPU에서 global load를 L1/texture cache 경로로 읽음
// #pragma unroll 4: inner loop를 펼쳐서 분기 오버헤드 감소
// block_size=128: Jetson Nano 단일 SM에서 최적 occupancy
// ═══════════════════════════════════════════════════════════
__global__ void score_all_kernel_v4(
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

  // 협력 적재: __ldg로 읽기 전용 캐시 활용
  for (int j = threadIdx.x; j < p; j += blockDim.x) {
    s_px[j] = __ldg(&px[j]);
    s_py[j] = __ldg(&py[j]);
  }
  __syncthreads();

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;

  int sum = 0;
  const int cxi = __ldg(&cx[i]);
  const int cyi = __ldg(&cy[i]);
  const unsigned int uw = static_cast<unsigned int>(w);
  const unsigned int uh = static_cast<unsigned int>(h);

  // unroll 4: 루프를 4개씩 묶어서 처리 → 분기/카운터 오버헤드 감소
  #pragma unroll 4
  for (int j = 0; j < p; ++j) {
    const unsigned int x = static_cast<unsigned int>(s_px[j] + cxi);
    const unsigned int y = static_cast<unsigned int>(s_py[j] + cyi);
    if (x < uw && y < uh)
      sum += __ldg(&grid[y * w + x]);
  }
  score[i] = static_cast<float>(sum) / (255.0f * static_cast<float>(p));
}

// ═══════════════════════════════════════════════════════════
// Persistent Buffer 관리
//
// 문제: Branch() 재귀 안에서 score_all이 수십~수백 번 반복 호출됨
//       cudaMalloc을 매번 하면 오버헤드 누적 → MatchWithWindow 전체 느려짐
// 해결: static 변수로 GPU 메모리를 프로그램 시작 시 1번만 할당,
//       grid(맵 데이터)는 변경 시에만 재업로드
// ═══════════════════════════════════════════════════════════
static unsigned char* d_grid_p  = nullptr;
static int*           d_px_p    = nullptr;
static int*           d_py_p    = nullptr;
static int*           d_cx_p    = nullptr;
static int*           d_cy_p    = nullptr;
static float*         d_score_p = nullptr;
static int            p_buf_n   = 0;
static int            p_buf_p   = 0;
static size_t         p_buf_g   = 0;

static void ensure_persistent_buf(size_t grid_size, int n, int p) {
  // 이미 충분한 크기로 할당되어 있으면 그냥 통과
  if (grid_size == p_buf_g && n <= p_buf_n && p <= p_buf_p) return;

  // 크기가 달라졌으면 기존 거 해제하고 새로 할당
  if (d_grid_p) {
    cudaFree(d_grid_p); cudaFree(d_px_p);    cudaFree(d_py_p);
    cudaFree(d_cx_p);   cudaFree(d_cy_p);    cudaFree(d_score_p);
  }
  cudaMalloc(&d_grid_p,  grid_size * sizeof(unsigned char));
  cudaMalloc(&d_px_p,    p * sizeof(int));
  cudaMalloc(&d_py_p,    p * sizeof(int));
  cudaMalloc(&d_cx_p,    n * sizeof(int));
  cudaMalloc(&d_cy_p,    n * sizeof(int));
  cudaMalloc(&d_score_p, n * sizeof(float));

  p_buf_g = grid_size;
  p_buf_n = n;
  p_buf_p = p;
}

// ═══════════════════════════════════════════════════════════
// score_all_gpu_v2: persistent buffer + v4 커널 (최종 최적 버전)
//
// 버전 교체 방법:
//   fast_matcher.cpp의 Score() 함수 안에서
//   score_all_gpu_v2(grid.cell, ..., &score); 한 줄만 바꾸면 됨
// ═══════════════════════════════════════════════════════════
void score_all_gpu_v2(const std::vector<unsigned char>& grid, const int w,
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

  // persistent buffer 확보 (처음만 cudaMalloc, 이후는 그냥 통과)
  ensure_persistent_buf(grid.size(), n, p);

  // grid는 변하지 않으면 재업로드 안 함 (persistent 핵심)
  static size_t last_grid_size = 0;
  if (grid.size() != last_grid_size) {
    cudaMemcpy(d_grid_p, grid.data(),
               grid.size() * sizeof(unsigned char), cudaMemcpyHostToDevice);
    last_grid_size = grid.size();
  }

  // 후보/스캔 데이터는 매 호출마다 다르므로 매번 업로드
  cudaMemcpy(d_px_p, px.data(), p * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_py_p, py.data(), p * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_cx_p, cx.data(), n * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_cy_p, cy.data(), n * sizeof(int), cudaMemcpyHostToDevice);

  // block_size=128: Jetson Nano 단일 SM에서 측정 결과 가장 빠름
  const int threads = 128;
  const int blocks  = (n + threads - 1) / threads;
  const size_t shared_mem = 2 * p * sizeof(int);

  auto t0 = std::chrono::high_resolution_clock::now();

  score_all_kernel_v4<<<blocks, threads, shared_mem>>>(
      d_grid_p, w, h, d_px_p, d_py_p, d_cx_p, d_cy_p, d_score_p, n, p);
  cudaDeviceSynchronize();

  cudaMemcpy(score->data(), d_score_p, n * sizeof(float), cudaMemcpyDeviceToHost);

  auto t1 = std::chrono::high_resolution_clock::now();
  auto us = std::chrono::duration_cast<std::chrono::microseconds>(t1-t0).count();
  std::cout << "[SCORE_ALL_gpu_v2_persist] candidates=" << n
            << ", scan_points=" << p
            << ", time=" << us << " us\n";
}

}  // namespace cartographer_parallel
