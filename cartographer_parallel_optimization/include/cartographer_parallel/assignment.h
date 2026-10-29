#ifndef CARTOGRAPHER_PARALLEL_ASSIGNMENT_H_
#define CARTOGRAPHER_PARALLEL_ASSIGNMENT_H_

#include <vector>

namespace cartographer_parallel {

// ───────────────────────────────────────────────
// CPU 버전들
// ───────────────────────────────────────────────

// baseline: 순차 이중 루프
void make_cand(int min_x, int max_x, int min_y, int max_y, int step,
               std::vector<int>* cx, std::vector<int>* cy);

void score_all(const std::vector<unsigned char>& grid, int w, int h,
               const std::vector<int>& px, const std::vector<int>& py,
               const std::vector<int>& cx, const std::vector<int>& cy,
               std::vector<float>* score);

// CPU 최적화: pointer caching + hoisting + unsigned 트릭 + OpenMP static
void score_all_cpu_opt(const std::vector<unsigned char>& grid, int w, int h,
                       const std::vector<int>& px, const std::vector<int>& py,
                       const std::vector<int>& cx, const std::vector<int>& cy,
                       std::vector<float>* score);

// CPU 단계별 버전들
void score_all_ptr(const std::vector<unsigned char>& grid, int w, int h,
                   const std::vector<int>& px, const std::vector<int>& py,
                   const std::vector<int>& cx, const std::vector<int>& cy,
                   std::vector<float>* score);

void score_all_hoist(const std::vector<unsigned char>& grid, int w, int h,
                     const std::vector<int>& px, const std::vector<int>& py,
                     const std::vector<int>& cx, const std::vector<int>& cy,
                     std::vector<float>* score);

void score_all_inv(const std::vector<unsigned char>& grid, int w, int h,
                   const std::vector<int>& px, const std::vector<int>& py,
                   const std::vector<int>& cx, const std::vector<int>& cy,
                   std::vector<float>* score);

void score_all_unsign(const std::vector<unsigned char>& grid, int w, int h,
                      const std::vector<int>& px, const std::vector<int>& py,
                      const std::vector<int>& cx, const std::vector<int>& cy,
                      std::vector<float>* score);

void score_all_omp_static(const std::vector<unsigned char>& grid, int w, int h,
                          const std::vector<int>& px, const std::vector<int>& py,
                          const std::vector<int>& cx, const std::vector<int>& cy,
                          std::vector<float>* score);

void score_all_omp_dynamic(const std::vector<unsigned char>& grid, int w, int h,
                           const std::vector<int>& px, const std::vector<int>& py,
                           const std::vector<int>& cx, const std::vector<int>& cy,
                           std::vector<float>* score);

// ───────────────────────────────────────────────
// GPU 버전들
// ───────────────────────────────────────────────

// GPU shmem (cudaMalloc 매번)
void score_all_gpu(const std::vector<unsigned char>& grid, const int w,
                   const int h, const std::vector<int>& px,
                   const std::vector<int>& py, const std::vector<int>& cx,
                   const std::vector<int>& cy,
                   std::vector<float>* const score);

// GPU persist + shmem + __ldg + unroll (최종 최적)
void score_all_gpu_v2(const std::vector<unsigned char>& grid, const int w,
                      const int h, const std::vector<int>& px,
                      const std::vector<int>& py, const std::vector<int>& cx,
                      const std::vector<int>& cy,
                      std::vector<float>* const score);

}  // namespace cartographer_parallel

#endif  // CARTOGRAPHER_PARALLEL_ASSIGNMENT_H_
