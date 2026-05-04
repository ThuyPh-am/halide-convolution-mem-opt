#pragma once
#include <Halide.h>

// output(n, k, y, x) = sum over r, s, c of: input(n, c, y*stride+r, x*stride+s) * weights(k, c, r, s)

struct ConvResult
{
    Halide::Func convolution;
    Halide::ImageParam input;
    Halide::ImageParam weights;
    Halide::RDom r;
    Halide::Var W_out, H_out, C_out;
};

ConvResult convo_base(int stride = 1, int pad = 0);