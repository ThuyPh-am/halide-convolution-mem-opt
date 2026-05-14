#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <utility>
#include <time.h>
#include <Halide.h>
#include "../config/layer_definitions.h"

using namespace Halide;

float random_generator();
std::pair<Buffer<float>, Buffer<float>> data_generating(const Convo& layer);