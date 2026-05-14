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

std::pair<Buffer<float>, Buffer<float>> data_generating(const Convo& layer){   
    srand(time(NULL)); 

    Buffer<float> in_buf(layer.W_in, layer.H_in, layer.C_in);
    Buffer<float> weights_buf(layer.K_w, layer.K_h, layer.C_in, layer.C_out);
    // generate input buffer , weight buffer
    in_buf.for_each_value([](float& v){v = random_generator();});
    weights_buf.for_each_value([](float& v){v = random_generator();});
    return {in_buf, weights_buf};
};
