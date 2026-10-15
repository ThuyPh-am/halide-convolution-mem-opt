#include <stdio.h>
#include <vector>
#include <string>
#include <iostream>
#include <algorithm>
#include <Halide.h>
#include "../config/layer_definitions.h"
#include "../generator/convolution_base.h"
#include "../runner/data_generator.h"
#include "../runner/perf_wrapper.h"
#include "../utils/csv_writer.h"

using namespace Halide;

static constexpr bool INSPECT = true;
static const std::string output_path = "../results/raw/rerun_results.csv";

// tile size------------------------------------------
// Choosen to fit a tile of (H_tile output rows) x (W_out) partial sums plus
// the corresponding input row into L2 (~512 KB on Cortex~A76)
static constexpr int H_TILE = 4;     // output rows per tile
static constexpr int C_OUT_TILE = 4; // output channels per tile (filter reuse)

void measure_layer_rs(const Convo &layer,
                      const std::string &model)
{

    auto [in_buff, weights_buff] = data_generating(layer);

    auto rs_result = convo_base(layer.stride, layer.pad);
    rs_result.input.set(in_buff);
    rs_result.weights.set(weights_buff);

    const int W_out_ext = layer.W_out;
    const int H_out_ext = layer.H_out;
    const int C_out_ext = layer.C_out;
    const int tile_H = std::min(H_TILE, H_out_ext);
    const int tile_C = std::min(C_OUT_TILE, C_out_ext);
    const int vec = std::min(8, W_out_ext);

    rs_result.convolution.compute_root();

    // Update reduction schedule-----------------------------------------
    // Originally RDom r is ordered (r[0] = kW, r[1] = kH, r[2] = C_in)
    // for RS schedule: the loop order is
    // Order: C_in (outermost) -> kH -> kW -> W_out -> H_inner -> C_inner (innermost)
    //
    // This way:
    //  * For a fixed (kH, kW) position the filter row element is reused across
    //      all output columns (W_out) and the H_TILE output rows -> filter-row stationary
    //  * The matching input row (stride*H _inner + kH, stride*W_out + kW)
    //      is accessed sequentially -> spatial locality
    //  * Partial sum (C_inner, H_inner, W_out) stay hot in L1 while the reduction sweeps all C_in and all filter rows

    // update stage: must split its own vars independently
    Var H_out_outer_u("H_out_outer_u"), H_out_inner_u("H_out_inner_u");
    Var C_out_outer_u("C_out_outer_u"), C_out_inner_u("C_out_inner_u");

    rs_result.convolution.update()
        .split(rs_result.H_out, H_out_outer_u, H_out_inner_u, tile_H, TailStrategy::GuardWithIf)
        .split(rs_result.C_out, C_out_outer_u, C_out_inner_u, tile_C, TailStrategy::GuardWithIf)
        .reorder(rs_result.W_out,
                 H_out_inner_u,
                 C_out_inner_u,
                 rs_result.r[0], // kW
                 rs_result.r[1], // kH
                 rs_result.r[2], // C_in
                 C_out_outer_u, H_out_outer_u)
        .vectorize(rs_result.W_out, vec, TailStrategy::GuardWithIf);
    // .parallel(H_out_outer_u);
    // .parallel(C_out_outer_u);

    if (INSPECT)
    {
        std::cout << "\n======RS loop nest for [" << layer.name << "]====\n";
        rs_result.convolution.print_loop_nest();
        std::cout << "============== \n\n";
    }

    rs_result.convolution.compile_jit(); // compile outside the counted window

    Buffer<float> output(W_out_ext, H_out_ext, C_out_ext);

    output.fill(0.0f); // fault output pages in before counting
    PerfResult rs_res = measure_convolution([&]()
                                            { rs_result.convolution.realize(output); });

    rs_res.schedule_name = "Row-stationary inspired";
    rs_res.model_name = model;
    rs_res.layer_name = layer.name;
    write_csv(output_path, rs_res);
    std::cout << "[" << layer.name << "] output: "
              << layer.W_out << "x" << layer.H_out << "x" << layer.C_out << "\n";
    std::cout << "======= end ======= \n\n";
}

// int main()
// {
//     try
//     {
//         std::cout << "Baseline schedule on Alex_conv2\n";
//         measure_layer_baseline(AlexNet::conv2, "AlexNet");
//         std::cout << "RS schedule on Alex_conv2\n";
//         measure_layer_rs(AlexNet::conv2, "AlexNet");
//     }
//     catch (const Halide::CompileError &e)
//     {
//         std::cerr << "Halide::CompileError " << e.what() << "\n";
//     }
// }