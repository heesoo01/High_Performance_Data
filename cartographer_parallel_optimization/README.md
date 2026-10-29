# Fast Correlative Scan Matcher — CPU/GPU 병렬화 최적화

> Inha University | 실시간 데이터 해석 | 심인욱 교수님  
> 12236649 채희수 | PA02 과제

---

## 프로젝트 개요

Google Cartographer의 **Fast Correlative Scan Matcher**를 Jetson Nano 환경에서 CPU 및 GPU 레벨로 고속화한 프로젝트입니다.  
라이다 스캔 데이터를 맵에 매칭하는 핵심 연산을 병렬화하여 실시간 처리 성능을 개선하였습니다.

---

## 실험 환경

| 항목 | 사양 |
|------|------|
| 실행 장비 | NVIDIA Jetson Nano (Maxwell, 128 CUDA cores, 1 SM) |
| OS | Ubuntu 18.04.6 LTS |
| ROS | Melodic |
| CUDA | 10.2.300 |
| 실행 환경 | Docker container |
| Map size | 467 × 314 cells (resolution 0.05 m/cell) |
| Candidates | 256 (coarse 단계) |
| Scan points | 1,081 |

---

## 파일 구조

```
cartographer_parallel/
├── src/
│   ├── fast_correlative_node.cpp   # ROS 노드 (건드리지 않음)
│   ├── fast_matcher.cpp            # MakeScans, MakeGridStack 최적화 포함
│   ├── score_all.cpp               # CPU 버전들 (baseline ~ cpu_opt)
│   ├── score_all_gpu.cu            # GPU shmem 버전
│   └── score_all_gpu_v2.cu         # GPU persist + __ldg + unroll (최종 최적 버전)
├── include/cartographer_parallel/
│   ├── fast_matcher.h
│   └── assignment.h
├── launch/
│   └── cartographer_parallel_with_bag.launch
└── CMakeLists.txt
```

---

## 고속화 대상 함수 및 결과 요약

### 전체 함수 검토

| 함수 | 역할 | 고속화 여부 | 이유 |
|------|------|------------|------|
| `MakeScans()` | 각도별 스캔 회전 변환 | ✅ CPU OMP | 각도 루프 완전 독립 연산 |
| `MakeBounds()` | 탐색 범위 계산 | ❌ Skip | 연산량 작음 |
| `MakeGridStack()` | 멀티레벨 max pooling | ✅ CPU OMP | 픽셀 단위 독립 연산 |
| `make_cand()` | 후보 좌표 생성 | ❌ Skip | push_back 구조상 병렬화 어려움 |
| `score_all()` | 후보-맵 매칭 점수 계산 | ✅ CPU OMP + GPU | 핵심 병목, N×P 이중 루프 |
| `Branch()` | Branch-and-bound 탐색 | ❌ Skip | 순차 의존성 존재 |
| `ToOut()` | 후보를 출력 pose로 변환 | ❌ Skip | 단순 변환, 연산량 무시 가능 |

---

## 1. score_all() 최적화

### 왜 이 함수인가
- candidates(256) × scan points(1081) = **약 28만 번 반복**하는 이중 루프
- candidate i의 점수 계산은 다른 candidate와 **완전히 독립적** → 병렬화 적합
- 전체 실행 시간의 대부분을 차지하는 핵심 병목

### CPU 단계별 최적화

| 버전 | 핵심 아이디어 | 평균 시간 (us) | Speedup |
|------|-------------|-------------|---------|
| baseline | 순차 이중 루프 | 11,158 | 1.00x |
| pointer caching | `vector.data()`로 raw pointer 캐싱 | 6,262 | 1.78x |
| cx/cy hoisting | inner loop 전 cx[i], cy[i] 지역변수 캐싱 | 5,919 | 1.88x |
| invariant hoisting | `1/(255*p)` 루프 밖 미리 계산 | 6,313 | 1.77x |
| unsigned 트릭 | 경계체크 4번 → unsigned 캐스팅으로 2번 | 5,164 | 2.16x |
| OMP static | `#pragma omp parallel for schedule(static)` | 1,487 | 7.50x |
| OMP dynamic | `#pragma omp parallel for schedule(dynamic,64)` | 1,557 | 7.17x |
| **CPU 누적 최적** | pointer + hoisting + unsigned + OMP static | 1,494 | **7.47x** |

### GPU 최적화

| 버전 | 핵심 아이디어 | 평균 시간 (us) | Speedup |
|------|-------------|-------------|---------|
| GPU shmem | px/py를 shared memory에 협력 적재 | 937 | 11.91x |
| GPU persist | shmem + cudaMalloc 제거 + grid 1회 업로드 | 921 | 12.11x |
| **GPU v4 (최종)** | persist + `__ldg` + `#pragma unroll 4` + block=128 | ~810 | **~13.8x** |

### 정확도 검증
모든 최적화 버전에서 `max_diff = 0.000000` — baseline과 완전히 동일한 결과 확인

---

## 2. MakeScans() 최적화

### 왜 이 함수인가
- 15개 각도 × 1081개 포인트 좌표 회전 변환
- 매 각도마다 cos/sin을 반복 계산하고 push_back으로 벡터 채우는 비효율 구조
- 각도 단위로 완전히 독립적 → OpenMP 적합

### 적용한 최적화
- **cos/sin 사전 계산**: 각도 루프 밖에서 `cos_table[]`, `sin_table[]` 미리 계산
- **pre-allocation**: `resize()`로 미리 공간 확보 → `push_back` 제거
- **OpenMP**: 각도 루프에 `#pragma omp parallel for schedule(static)`

### 결과

| 버전 | 시간 (us) | Speedup |
|------|---------|---------|
| baseline | 3,404 | 1.00x |
| OMP + cos/sin 사전계산 + pre-alloc | 2,569 | **1.32x** |

---

## 3. MakeGridStack() 최적화

### 왜 이 함수인가
- 맵 전체 픽셀(467×314 = 약 14만개)에 대해 윈도우 max pooling 계산
- 픽셀 (x,y)의 계산은 다른 픽셀과 완전히 독립적
- branch_depth에 따라 연산량 달라짐

### 적용한 최적화
픽셀 루프 위에 한 줄 추가:
```cpp
#pragma omp parallel for schedule(static)
for (int y = 0; y < h_; ++y) { ... }
```

### 결과

| matcher | branch_depth | baseline (us) | OMP (us) | Speedup |
|---------|-------------|--------------|---------|---------|
| matcher_ | 4 | 373,163 | 181,703 | 2.05x |
| initial_matcher_ | 3 | 90,934 | 59,135 | 1.54x |
| initial_refine_matcher_ | 2 | 26,471 | 17,269 | 1.53x |
| **평균** | - | ~163,523 | ~86,036 | **~1.71x** |

---

## 버전 교체 방법

### score_all() 버전 바꾸기
`fast_matcher.cpp`의 `Score()` 함수 안에서 **이 한 줄만** 교체:

```cpp
// baseline
score_all(grid.cell, grid.w, grid.h, scans[s].x, scans[s].y, cx, cy, &score);

// CPU 최적
score_all_cpu_opt(grid.cell, grid.w, grid.h, scans[s].x, scans[s].y, cx, cy, &score);

// GPU persist (최종 최적)
cartographer_parallel::score_all_gpu_v2(grid.cell, grid.w, grid.h, scans[s].x, scans[s].y, cx, cy, &score);
```

### MakeScans() 버전 바꾸기
`fast_matcher.cpp`의 `MatchWithWindow()` 안에서:

```cpp
// baseline
const std::vector<Scan> scans = MakeScans(xs, ys, init, &num_ang, &step);

// 최적화 버전
const std::vector<Scan> scans = MakeScansOpt(xs, ys, init, &num_ang, &step);
```

### MakeGridStack() 버전 바꾸기
`fast_matcher.cpp`의 `MakeGridStack()` 함수 안에서:

```cpp
// baseline: 주석 처리
//#pragma omp parallel for schedule(static)

// 최적화: 주석 해제
#pragma omp parallel for schedule(static)
```

---

## 빌드 및 실행

```bash
# Docker 진입
sudo docker start "student_00" && sudo docker exec -it "student_00" /bin/bash

# 파일 복사 및 빌드
mkdir -p ~/catkin_ws/src
cd ~/catkin_ws/src
cp -r /data/cartographer_parallel .
cd ~/catkin_ws
catkin_make
source devel/setup.bash

# 실행
roslaunch cartographer_parallel cartographer_parallel_with_bag.launch ns:="student_00"
```

---

## 핵심 개념 정리

### 왜 이 함수들이 병렬화 가능한가
세 함수 모두 공통 원리: **각 반복 항목이 서로 독립적 (embarrassingly parallel)**
- `score_all`: candidate i는 score[i]에만 쓰고, 다른 candidate와 데이터 교환 없음
- `MakeScans`: 각도 s의 회전 변환은 다른 각도와 무관
- `MakeGridStack`: 픽셀 (x,y)의 max pooling은 다른 픽셀에 영향 없음

### Persistent Buffer가 왜 중요한가
GPU 함수는 `Branch()` 재귀 안에서 수십~수백 번 반복 호출됨.  
`cudaMalloc`을 매번 하면 오버헤드 누적 → 프로그램 시작 시 1회만 할당해서 재사용.  
단일 호출 시간은 비슷해도, 전체 프레임 시간(`MatchWithWindow`)에서 극적인 차이 발생:
- GPU shmem (persistent 없음): 210,272 us
- GPU persist: **29,215 us** (7배 이상 차이)

---

## 전체 비교 결과

| 함수 | baseline | 최적 버전 | 최적 시간 | Speedup |
|------|---------|---------|---------|---------|
| score_all() | 11,158 us | GPU v4 (persist+ldg+unroll) | ~810 us | **~13.8x** |
| MakeScans() | 3,404 us | OMP + 사전계산 | 2,569 us | 1.32x |
| MakeGridStack() | ~90,934 us | OMP | ~59,135 us | 1.54x |
| MatchWithWindow 전체 | 194,800 us | GPU persist | 29,215 us | **6.67x** |

