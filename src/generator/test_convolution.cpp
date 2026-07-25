#include <iostream>
#include <Halide.h>
#include "convolution_base.h"

using namespace Halide;

int main()
{

    // === Realize on a small test case ===
    // 4x4 input, 1 channel, 2x2 kernel, 1 output filter
    Buffer<float> in_buf(5, 5, 3);
    Buffer<float> w_buf(3, 3, 3, 2);

    for (int c = 0; c < 3; c++)
        for (int h = 0; h < 5; h++)
            for (int w = 0; w < 5; w++)
                in_buf(w, h, c) = 1.0f;

    for (int co = 0; co < 2; co++)
        for (int ci = 0; ci < 3; ci++)
            for (int kh = 0; kh < 3; kh++)
                for (int kw = 0; kw < 3; kw++)
                    w_buf(kw, kh, ci, co) = 1.0f;

    // run baseline
    auto base = convo_base(1, 0);
    base.input.set(in_buf);
    base.weights.set(w_buf);
    base.convolution.compute_root();
    Buffer<float> base_out = base.convolution.realize({3, 3, 2});

    // run rs
    auto rs = convo_base(1, 0);
    rs.input.set(in_buf);
    rs.weights.set(w_buf);

    static constexpr int H_TILE = 2;
    static constexpr int C_OUT_TILE = 2;
    const int tile_H = std::min(H_TILE, 3);
    const int tile_C = std::min(C_OUT_TILE, 2);
    const int vec = std::min(8, 3);

    Var H_out_outer_u("H_out_outer_u"), H_out_inner_u("H_out_inner_u");
    Var C_out_outer_u("C_out_outer_u"), C_out_inner_u("C_out_inner_u");

    rs.convolution.compute_root();
    rs.convolution.update()
        .split(rs.H_out, H_out_outer_u, H_out_inner_u, tile_H, TailStrategy::GuardWithIf)
        .split(rs.C_out, C_out_outer_u, C_out_inner_u, tile_C, TailStrategy::GuardWithIf)
        .reorder(rs.W_out,
                 H_out_inner_u,
                 C_out_inner_u,
                 rs.r[0], rs.r[1], rs.r[2],
                 C_out_outer_u, H_out_outer_u)
        .vectorize(rs.W_out, vec, TailStrategy::GuardWithIf);

    rs.convolution.print_loop_nest();
    Buffer<float> rs_out = rs.convolution.realize({3, 3, 2});

    // compare
    float max_diff = 0.0f;
    bool correct = true;
    for (int c = 0; c < 2; c++)
        for (int h = 0; h < 3; h++)
            for (int w = 0; w < 3; w++)
            {
                float diff = std::abs(base_out(w, h, c) - rs_out(w, h, c));
                if (diff > max_diff)
                    max_diff = diff;
                if (diff > 1e-3f)
                {
                    std::cout << "MISMATCH at (" << w << "," << h << "," << c << "): "
                              << "base=" << base_out(w, h, c)
                              << " rs=" << rs_out(w, h, c) << "\n";
                    correct = false;
                }
            }

    // with all-ones input and weights, every output = 3*3*3 = 27
    std::cout << "Expected all outputs = 27.0\n";
    std::cout << "base(0,0,0) = " << base_out(0, 0, 0) << "\n";
    std::cout << "rs(0,0,0)   = " << rs_out(0, 0, 0) << "\n";
    std::cout << (correct ? "CORRECT" : "WRONG")
              << " max_diff=" << max_diff << "\n";
    return correct ? 0 : 1;
}
