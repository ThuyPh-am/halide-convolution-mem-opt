#pragma once
#include <functional>
#include <string>

struct PerfResult {
    std::string model_name;
    std::string layer_name;
    std::string schedule_name;
    long long llc_misses; //l3D_CACHE REFILL - primary DRAM proxy
    long long llc_accesses; // l3D_CACHE_RD - for mixx-rate calculation
    long long l2_refill;
    long long l2_read;
    long long l2_write;
};

PerfResult measure_convolution(std::function<void()>conv_fd);