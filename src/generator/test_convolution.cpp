#include <iostream>
#include <Halide.h>
#include "convolution_base.h"

using namespace Halide;

int main()
{
    ConvResult res = convo_base(1, 0);

    // === Realize on a small test case ===
    // 4x4 input, 1 channel, 2x2 kernel, 1 output filter
    Buffer<float> in_buf(4, 4, 1);
    Buffer<float> w_buf(2, 2, 1, 1);

    float in_data[4][4] = {
        {1, 2, 3, 4},
        {4, 5, 6, 7},
        {2, 2, 3, 4},
        {1, 4, 5, 7},
    };

    const int W_out_ext = 3;
    const int H_out_ext = 3;
    const int C_out_ext = 1;
    static constexpr int H_TILE = 2;
    static constexpr int C_OUT_TILE = 2;
    const int tile_H = std::min(H_TILE, H_out_ext);
    const int tile_C = std::min(C_OUT_TILE, C_out_ext);
    const int vec = std::min(8, W_out_ext);
    for (int h = 0; h < 4; h++)
    {
        for (int w = 0; w < 4; w++)
        {
            in_buf(w, h, 0) = in_data[h][w];
        }
    }

    // kernel [[1,2],[0,5]]
    w_buf(0, 0, 0, 0) = 1;
    w_buf(1, 0, 0, 0) = 2;
    w_buf(0, 1, 0, 0) = 0;
    w_buf(1, 1, 0, 0) = 5;

    res.input.set(in_buf);
    res.weights.set(w_buf);

    Var H_out_outer_u("H_out_outer_u"), H_out_inner_u("H_out_inner_u");
    Var C_out_outer_u("C_out_outer_u"), C_out_inner_u("C_out_inner_u");

    // output is 3x3 (4-2+1) for stride = 1, no padding
    res.convolution.compute_root();
    res.convolution.update()
        .split(res.H_out, H_out_outer_u, H_out_inner_u, tile_H, TailStrategy::GuardWithIf)
        .split(res.C_out, C_out_outer_u, C_out_inner_u, tile_C, TailStrategy::GuardWithIf)
        .reorder(res.W_out,
                 H_out_inner_u,
                 C_out_inner_u,
                 res.r[0], // kW
                 res.r[1], // kH
                 res.r[2], // C_in
                 H_out_outer_u, C_out_outer_u)
        .vectorize(res.W_out, vec, TailStrategy::GuardWithIf)
        .parallel(C_out_outer_u);
    res.convolution.print_loop_nest();
    Buffer<float> output = res.convolution.realize({3, 3, 1});

    std::cout << "Output (3x3):" << std::endl;
    for (int h = 0; h < 3; h++)
    {
        for (int w = 0; w < 3; w++)
        {
            std::cout << output(w, h, 0) << "\t";
        }
        std::cout << std::endl;
    }
}
