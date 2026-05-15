cmake_minimum_required(VERSION 3.16)
project(halide_proj)

set(CMAKE_CXX_STANDARD 17)

# Halide paths
set(HALIDE_ROOT $ENV{HALIDE_ROOT})
find_package(Halide REQUIRED HINTS ${HALIDE_ROOT}/lib/cmake/Halide)

# Include config headers
include_directories(${CMAKE_SOURCE_DIR}/src/config)

# shared library — Halide::Halide is PUBLIC so all executables inherit it
add_library(conv_base STATIC src/generator/convolution_base.cpp)
target_link_libraries(conv_base PUBLIC Halide::Halide)
target_include_directories(conv_base PUBLIC src/generator)

add_library(data_generator STATIC src/runner/data_generator.cpp)
target_link_libraries(data_generator PUBLIC Halide::Halide)
target_include_directories(data_generator PUBLIC src/runner)

add_library(perf_wrapper STATIC src/runner/perf_wrapper.cpp)
target_include_directories(perf_wrapper PUBLIC src/runner)

add_library(csv_writer STATIC src/utils/csv_writer.cpp)
target_include_directories(csv_writer PUBLIC src/utils)

# executables
add_executable(run_experiment
    src/runner/main.cpp
    src/generator/schedule_baseline.cpp
    src/generator/schedule_rs_inspired.cpp
    src/generator/schedule_ws_inspired.cpp
    src/generator/schedule_os_inspired.cpp
)

target_link_libraries(run_experiment
    PRIVATE
        conv_base
        data_generator
        perf_wrapper
        csv_writer
)

#  sanity check
add_executable(test_convolution src/generator/test_convolution.cpp)
target_link_libraries(test_convolution PRIVATE conv_base)
