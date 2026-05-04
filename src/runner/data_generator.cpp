#include <stdio.h>
#include <stdlib.h>
#include <utility>
#include <time.h>
#include <Halide.h>
#include "../config/layer_definitions.h"

using namespace Halide;

float random_generator()
{
    return (float)rand() / RAND_MAX;
}

Buffer<float> calculate_output(const Convo& layer)
{
    int W_out = out_dim(layer.W_in, layer.pad, layer.K_w, layer.stride);
    int H_out = out_dim(layer.H_in, layer.pad, layer.K_h, layer.stride);
    return Buffer<float> (W_out, H_out, layer.C_out);
}

std::pair<Buffer<float>, Buffer<float>> data_generating(const Convo& layer){   
    srand(time(NULL)); 

    Buffer<float> in_buf(layer.W_in, layer.H_in, layer.C_in);
    Buffer<float> weights_buf(layer.K_w, layer.K_h, layer.C_in, layer.C_out);
    // generate input buffer , weight buffer
    in_buf.for_each_value([](float& v){v = random_generator();});
    weights_buf.for_each_value([](float& v){v = random_generator();});
    return {in_buf, weights_buf};
};
