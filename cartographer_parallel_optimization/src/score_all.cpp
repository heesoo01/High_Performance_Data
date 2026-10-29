#include <chrono>
#include <iostream>
#include "cartographer_parallel/assignment.h"
#include <algorithm>

namespace cartographer_parallel {

// ═══════════════════════════════════════════════════════════
// make_cand: 후보 좌표 생성
// ═══════════════════════════════════════════════════════════
void make_cand(const int min_x, const int max_x, const int min_y,
               const int max_y, const int step, std::vector<int>* const cx,
               std::vector<int>* const cy) {
  if (cx == nullptr || cy == nullptr || step <= 0) return;
  for (int x = min_x; x <= max_x; x += step) {
    for (int y = min_y; y <= max_y; y += step) {
      cx->push_back(x);
      cy->push_back(y);
    }
  }
}

// ═══════════════════════════════════════════════════════════
// baseline: 순차 이중 루프
// ═══════════════════════════════════════════════════════════
void score_all(const std::vector<unsigned char>& grid, const int w,
               const int h, const std::vector<int>& px,
               const std::vector<int>& py, const std::vector<int>& cx,
               const std::vector<int>& cy, std::vector<float>* const score) {
  if (score == nullptr) return;
  const int n = std::min(cx.size(), cy.size());
  const int p = std::min(px.size(), py.size());
  score->assign(n, 0.0f);
  if (w <= 0 || h <= 0 || p == 0 || grid.size() < static_cast<size_t>(w * h))
    return;
  for (int i = 0; i < n; ++i) {
    int sum = 0;
    for (int j = 0; j < p; ++j) {
      const int x = px[j] + cx[i];
      const int y = py[j] + cy[i];
      if (x >= 0 && x < w && y >= 0 && y < h)
        sum += grid[y * w + x];
    }
    (*score)[i] = static_cast<float>(sum) / (255.0f * static_cast<float>(p));
  }
}

// ═══════════════════════════════════════════════════════════
// pointer caching: vector.data()로 raw pointer 캐싱
// ═══════════════════════════════════════════════════════════
void score_all_ptr(const std::vector<unsigned char>& grid, const int w,
                   const int h, const std::vector<int>& px,
                   const std::vector<int>& py, const std::vector<int>& cx,
                   const std::vector<int>& cy, std::vector<float>* const score) {
  if (score == nullptr) return;
  const int n = std::min(cx.size(), cy.size());
  const int p = std::min(px.size(), py.size());
  score->assign(n, 0.0f);
  if (w <= 0 || h <= 0 || p == 0 || grid.size() < static_cast<size_t>(w * h))
    return;
  const unsigned char* g = grid.data();
  const int* px_ = px.data();
  const int* py_ = py.data();
  const int* cx_ = cx.data();
  const int* cy_ = cy.data();
  float* s = score->data();
  for (int i = 0; i < n; ++i) {
    int sum = 0;
    for (int j = 0; j < p; ++j) {
      const int x = px_[j] + cx_[i];
      const int y = py_[j] + cy_[i];
      if (x >= 0 && x < w && y >= 0 && y < h)
        sum += g[y * w + x];
    }
    s[i] = static_cast<float>(sum) / (255.0f * static_cast<float>(p));
  }
}

// ═══════════════════════════════════════════════════════════
// cx/cy hoisting: inner loop 전 지역변수로 캐싱
// ═══════════════════════════════════════════════════════════
void score_all_hoist(const std::vector<unsigned char>& grid, const int w,
                     const int h, const std::vector<int>& px,
                     const std::vector<int>& py, const std::vector<int>& cx,
                     const std::vector<int>& cy, std::vector<float>* const score) {
  if (score == nullptr) return;
  const int n = std::min(cx.size(), cy.size());
  const int p = std::min(px.size(), py.size());
  score->assign(n, 0.0f);
  if (w <= 0 || h <= 0 || p == 0 || grid.size() < static_cast<size_t>(w * h))
    return;
  const unsigned char* g = grid.data();
  const int* px_ = px.data();
  const int* py_ = py.data();
  const int* cx_ = cx.data();
  const int* cy_ = cy.data();
  float* s = score->data();
  for (int i = 0; i < n; ++i) {
    int sum = 0;
    const int cxi = cx_[i];   // ← hoisting: inner loop 밖으로 꺼냄
    const int cyi = cy_[i];
    for (int j = 0; j < p; ++j) {
      const int x = px_[j] + cxi;
      const int y = py_[j] + cyi;
      if (x >= 0 && x < w && y >= 0 && y < h)
        sum += g[y * w + x];
    }
    s[i] = static_cast<float>(sum) / (255.0f * static_cast<float>(p));
  }
}

// ═══════════════════════════════════════════════════════════
// invariant hoisting: 1/(255*p) 루프 밖에서 미리 계산
// ═══════════════════════════════════════════════════════════
void score_all_inv(const std::vector<unsigned char>& grid, const int w,
                   const int h, const std::vector<int>& px,
                   const std::vector<int>& py, const std::vector<int>& cx,
                   const std::vector<int>& cy, std::vector<float>* const score) {
  if (score == nullptr) return;
  const int n = std::min(cx.size(), cy.size());
  const int p = std::min(px.size(), py.size());
  score->assign(n, 0.0f);
  if (w <= 0 || h <= 0 || p == 0 || grid.size() < static_cast<size_t>(w * h))
    return;
  const unsigned char* g = grid.data();
  const int* px_ = px.data();
  const int* py_ = py.data();
  const int* cx_ = cx.data();
  const int* cy_ = cy.data();
  float* s = score->data();
  const float inv = 1.0f / (255.0f * static_cast<float>(p));  // ← 루프 밖으로
  for (int i = 0; i < n; ++i) {
    int sum = 0;
    const int cxi = cx_[i];
    const int cyi = cy_[i];
    for (int j = 0; j < p; ++j) {
      const int x = px_[j] + cxi;
      const int y = py_[j] + cyi;
      if (x >= 0 && x < w && y >= 0 && y < h)
        sum += g[y * w + x];
    }
    s[i] = static_cast<float>(sum) * inv;  // 나눗셈 → 곱셈
  }
}

// ═══════════════════════════════════════════════════════════
// unsigned 비교 트릭: 경계체크 4번 → 2번
// 음수를 unsigned로 캐스팅하면 매우 큰 양수가 되어 자동으로 걸러짐
// ═══════════════════════════════════════════════════════════
void score_all_unsign(const std::vector<unsigned char>& grid, const int w,
                      const int h, const std::vector<int>& px,
                      const std::vector<int>& py, const std::vector<int>& cx,
                      const std::vector<int>& cy, std::vector<float>* const score) {
  if (score == nullptr) return;
  const int n = std::min(cx.size(), cy.size());
  const int p = std::min(px.size(), py.size());
  score->assign(n, 0.0f);
  if (w <= 0 || h <= 0 || p == 0 || grid.size() < static_cast<size_t>(w * h))
    return;
  const unsigned char* g = grid.data();
  const int* px_ = px.data();
  const int* py_ = py.data();
  const int* cx_ = cx.data();
  const int* cy_ = cy.data();
  float* s = score->data();
  const float inv = 1.0f / (255.0f * static_cast<float>(p));
  const unsigned int uw = static_cast<unsigned int>(w);
  const unsigned int uh = static_cast<unsigned int>(h);
  for (int i = 0; i < n; ++i) {
    int sum = 0;
    const int cxi = cx_[i];
    const int cyi = cy_[i];
    for (int j = 0; j < p; ++j) {
      const unsigned int x = static_cast<unsigned int>(px_[j] + cxi);
      const unsigned int y = static_cast<unsigned int>(py_[j] + cyi);
      if (x < uw && y < uh)   // ← 4비교 → 2비교
        sum += g[y * w + x];
    }
    s[i] = static_cast<float>(sum) * inv;
  }
}

// ═══════════════════════════════════════════════════════════
// OpenMP static: candidate 루프를 4코어에 균등 분배
// schedule(static): 후보마다 연산량이 같아서 균등 분배가 최적
// ═══════════════════════════════════════════════════════════
void score_all_omp_static(const std::vector<unsigned char>& grid, const int w,
                          const int h, const std::vector<int>& px,
                          const std::vector<int>& py, const std::vector<int>& cx,
                          const std::vector<int>& cy, std::vector<float>* const score) {
  if (score == nullptr) return;
  const int n = std::min(cx.size(), cy.size());
  const int p = std::min(px.size(), py.size());
  score->assign(n, 0.0f);
  if (w <= 0 || h <= 0 || p == 0 || grid.size() < static_cast<size_t>(w * h))
    return;
  const unsigned char* g = grid.data();
  const int* px_ = px.data();
  const int* py_ = py.data();
  const int* cx_ = cx.data();
  const int* cy_ = cy.data();
  float* s = score->data();
  const float inv = 1.0f / (255.0f * static_cast<float>(p));
  const unsigned int uw = static_cast<unsigned int>(w);
  const unsigned int uh = static_cast<unsigned int>(h);
#pragma omp parallel for schedule(static)
  for (int i = 0; i < n; ++i) {
    int sum = 0;
    const int cxi = cx_[i];
    const int cyi = cy_[i];
    for (int j = 0; j < p; ++j) {
      const unsigned int x = static_cast<unsigned int>(px_[j] + cxi);
      const unsigned int y = static_cast<unsigned int>(py_[j] + cyi);
      if (x < uw && y < uh)
        sum += g[y * w + x];
    }
    s[i] = static_cast<float>(sum) * inv;
  }
}

// ═══════════════════════════════════════════════════════════
// OpenMP dynamic: chunk=64로 부하 균형 + 스케줄링 오버헤드 절충
// ═══════════════════════════════════════════════════════════
void score_all_omp_dynamic(const std::vector<unsigned char>& grid, const int w,
                           const int h, const std::vector<int>& px,
                           const std::vector<int>& py, const std::vector<int>& cx,
                           const std::vector<int>& cy, std::vector<float>* const score) {
  if (score == nullptr) return;
  const int n = std::min(cx.size(), cy.size());
  const int p = std::min(px.size(), py.size());
  score->assign(n, 0.0f);
  if (w <= 0 || h <= 0 || p == 0 || grid.size() < static_cast<size_t>(w * h))
    return;
  const unsigned char* g = grid.data();
  const int* px_ = px.data();
  const int* py_ = py.data();
  const int* cx_ = cx.data();
  const int* cy_ = cy.data();
  float* s = score->data();
  const float inv = 1.0f / (255.0f * static_cast<float>(p));
  const unsigned int uw = static_cast<unsigned int>(w);
  const unsigned int uh = static_cast<unsigned int>(h);
#pragma omp parallel for schedule(dynamic, 64)
  for (int i = 0; i < n; ++i) {
    int sum = 0;
    const int cxi = cx_[i];
    const int cyi = cy_[i];
    for (int j = 0; j < p; ++j) {
      const unsigned int x = static_cast<unsigned int>(px_[j] + cxi);
      const unsigned int y = static_cast<unsigned int>(py_[j] + cyi);
      if (x < uw && y < uh)
        sum += g[y * w + x];
    }
    s[i] = static_cast<float>(sum) * inv;
  }
}

// ═══════════════════════════════════════════════════════════
// CPU 누적 최적 버전: pointer + hoisting + unsigned + OMP static
// ═══════════════════════════════════════════════════════════
void score_all_cpu_opt(const std::vector<unsigned char>& grid, const int w,
                       const int h, const std::vector<int>& px,
                       const std::vector<int>& py, const std::vector<int>& cx,
                       const std::vector<int>& cy, std::vector<float>* const score) {
  if (score == nullptr) return;
  const int n = std::min(cx.size(), cy.size());
  const int p = std::min(px.size(), py.size());
  score->assign(n, 0.0f);
  if (w <= 0 || h <= 0 || p == 0 || grid.size() < static_cast<size_t>(w * h))
    return;
  const unsigned char* g = grid.data();
  const int* px_ = px.data();
  const int* py_ = py.data();
  const int* cx_ = cx.data();
  const int* cy_ = cy.data();
  float* s = score->data();
  const float inv = 1.0f / (255.0f * static_cast<float>(p));
  const unsigned int uw = static_cast<unsigned int>(w);
  const unsigned int uh = static_cast<unsigned int>(h);
#pragma omp parallel for schedule(static)
  for (int i = 0; i < n; ++i) {
    int sum = 0;
    const int cxi = cx_[i];
    const int cyi = cy_[i];
    for (int j = 0; j < p; ++j) {
      const unsigned int x = static_cast<unsigned int>(px_[j] + cxi);
      const unsigned int y = static_cast<unsigned int>(py_[j] + cyi);
      if (x < uw && y < uh)
        sum += g[y * w + x];
    }
    s[i] = static_cast<float>(sum) * inv;
  }
}

}  // namespace cartographer_parallel
