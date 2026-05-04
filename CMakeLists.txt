cmake_minimum_required(VERSION 3.16)
project(halide_proj)

set(CMAKE_CXX_STANDARD 17)

# Halide paths
set(HALIDE_ROOT $ENV{HALIDE_ROOT})
find_package(Halide REQUIRED HINTS ${HALIDE_ROOT}/lib/cmake/Halide)

# Include your config headers
include_directories(${CMAKE_SOURCE_DIR}/src/config)

# shared library — Halide::Halide is PUBLIC so all executables inherit it
add_library(conv_base STATIC src/generator/convolution_base.cpp)
target_link_libraries(conv_base PUBLIC Halide::Halide)
target_include_directories(conv_base PUBLIC src/generator)

add_library(perf_wrapper STATIC src/runner/perf_wrapper.cpp)
target_include_directories(perf_wrapper PUBLIC src/runner)

add_library(csv_writer STATIC src/utils/csv_writer.cpp)
target_include_directories(csv_writer PUBLIC src/utils)
# executables
add_executable(test_convolution  src/generator/test_convolution.cpp)
add_executable(schedule_baseline src/generator/schedule_baseline.cpp)
add_executable(schedule_rs       src/generator/schedule_rs_inspired.cpp)
add_executable(schedule_ws       src/generator/schedule_ws_inspired.cpp)
add_executable(schedule_os       src/generator/schedule_os_inspired.cpp)

# Link Halide to all targets
foreach(target test_convolution schedule_baseline schedule_rs schedule_ws schedule_os)
    target_link_libraries(${target} conv_base perf_wrapper csv_writer)
endforeach()