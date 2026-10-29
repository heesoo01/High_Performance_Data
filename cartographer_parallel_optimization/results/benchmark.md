# 벤치마크 측정 결과

> Jetson Nano | CUDA 10.2 | candidates=256 | scan_points=1,081  
> 측정 방법: `std::chrono::high_resolution_clock`  
> 정확도 검증: 모든 버전에서 `max_diff = 0.000000` (baseline과 완전 동일)

---

## 1. score_all() — candidates=256, scan_points=1,081

| 버전 | 평균 시간 (us) | Speedup | MatchWithWindow (us) |
|------|-------------|---------|---------------------|
| baseline | 11,158 | 1.00x | 194,800 |
| pointer caching | 6,262 | 1.78x | 114,415 |
| cx/cy hoisting | 5,919 | 1.88x | 110,127 |
| invariant hoisting | 6,313 | 1.77x | 120,667 |
| unsigned 트릭 | 5,164 | 2.16x | 96,905 |
| OMP static | 1,487 | 7.50x | 72,471 |
| OMP dynamic | 1,557 | 7.17x | 54,748 |
| CPU 누적 최적 | 1,494 | 7.47x | 52,424 |
| GPU shmem | 937 | 11.91x | 210,272 |
| GPU persist (v3 커널) | 921 | 12.11x | 29,215 |
| **GPU v4 (ldg+unroll+b128)** | **~810** | **~13.8x** | **~27,000** |

### GPU block size 튜닝 결과 (v4 커널 기준)

| block size | 평균 시간 (us) | 비고 |
|-----------|-------------|------|
| 128 | ~810 | ✅ 최적 |
| 256 | ~820 | - |
| 512 | ~810 | 128과 유사 |

### 분석
- 단일 기법 중 **unsigned 트릭**이 2.16x로 가장 효과적 (inner loop 분기 절반으로 감소)
- **OMP static**이 dynamic보다 약간 우세 (256개 후보를 4코어에 64개씩 균등 분배)
- **GPU shmem**은 단일 함수 speedup은 높으나, persistent 없으면 `MatchWithWindow` 전체 시간이 오히려 증가 (Branch 재귀 호출 누적)
- **GPU persist**가 `MatchWithWindow` 전체에서 최종 winner (29,215 us vs cpu_opt 52,424 us)

---

## 2. MakeScans() — scan_count=15, scan_points=1,081

| 버전 | 시간 (us) | Speedup |
|------|---------|---------|
| baseline | 3,404 | 1.00x |
| OMP + cos/sin 사전계산 + pre-alloc | 2,569 | **1.32x** |

### 분석
- scan_count=15로 4코어 분배 시 코어당 3~4개만 처리 → 병렬화 이득 제한적
- cos/sin 사전계산 + pre-allocation 효과가 OMP와 합쳐져서 1.32x 달성

---

## 3. MakeGridStack() — 467×314 pixels, branch_depth별

| matcher | branch_depth | baseline (us) | OMP (us) | Speedup |
|---------|-------------|--------------|---------|---------|
| matcher_ | 4 | 373,163 | 181,703 | 2.05x |
| initial_matcher_ | 3 | 90,934 | 59,135 | 1.54x |
| initial_refine_matcher_ | 2 | 26,471 | 17,269 | 1.53x |
| **평균** | - | ~163,523 | ~86,036 | **~1.71x** |

### 분석
- 픽셀 수(14만)가 충분히 많아 OMP 효과가 잘 나타남
- branch_depth가 클수록 연산량 많아서 병렬화 이득도 커짐 (depth=4에서 2.05x)

---

## 최종 종합

| 함수 | baseline | 최적 버전 | 최적 시간 | Speedup |
|------|---------|---------|---------|---------|
| score_all() | 11,158 us | GPU v4 (persist+ldg+unroll+b128) | ~810 us | **~13.8x** |
| MakeScans() | 3,404 us | OMP + 사전계산 + pre-alloc | 2,569 us | 1.32x |
| MakeGridStack() | ~90,934 us | OMP | ~59,135 us | 1.54x |
| MatchWithWindow 전체 | 194,800 us | GPU persist | 29,215 us | **6.67x** |
