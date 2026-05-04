#pragma once
#include <string>
#include "../runner/perf_wrapper.h"

void write_csv(const std::string &output_path, const PerfResult &layer_result);