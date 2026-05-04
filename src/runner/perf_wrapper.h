#pragma once
#include <functional>
#include <string>

struct PerfResult {
    std::string model_name;
    std::string layer_name;
    std::string schedule_name;
    long long llc_misses;
};

PerfResult measure_convolution(std::function<void()>conv_fd);