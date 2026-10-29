cmake_minimum_required(VERSION 3.0.2)
project(cartographer_parallel)

option(BUILD_CUDA_TASK "Build optional CUDA assignment library" OFF)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_msgs
  sensor_msgs
  geometry_msgs
  nav_msgs
  visualization_msgs
)
find_package(OpenMP REQUIRED)
enable_language(CUDA)

# Jetson Nano: Maxwell 아키텍처 (compute capability 5.3)
set(CMAKE_CUDA_ARCHITECTURES 53)

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES assignment_cpu_lib fast_matcher_lib
  CATKIN_DEPENDS roscpp std_msgs sensor_msgs geometry_msgs nav_msgs visualization_msgs
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
)

# CPU 버전 라이브러리 (score_all 단계별 버전들)
add_library(assignment_cpu_lib
  src/score_all.cpp
)
target_link_libraries(assignment_cpu_lib ${catkin_LIBRARIES})

# 전체 matcher 라이브러리 (fast_matcher + GPU 커널 포함)
add_library(fast_matcher_lib
  src/fast_matcher.cpp
  src/score_all.cpp
  src/score_all_gpu.cu       # GPU shmem 버전
  src/score_all_gpu_v2.cu    # GPU persist + ldg + unroll (최종 최적)
)
target_link_libraries(fast_matcher_lib
  assignment_cpu_lib
  ${catkin_LIBRARIES}
  OpenMP::OpenMP_CXX
)
set_target_properties(fast_matcher_lib PROPERTIES
  CUDA_ARCHITECTURES "53"
)

# ROS 노드 실행파일
add_executable(fast_correlative_node src/fast_correlative_node.cpp)
target_link_libraries(fast_correlative_node fast_matcher_lib ${catkin_LIBRARIES})

install(TARGETS
  assignment_cpu_lib
  fast_matcher_lib
  fast_correlative_node
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch maps bags
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
