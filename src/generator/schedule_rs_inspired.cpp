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

static const std::string output_path = "../results/raw/raw_results.csv";

// tile size------------------------------------------
// Choosen to fit a tile of (H_tile output rows) x (W_out) partial sums plus
// the corresponding input row into L2 (~512 KB on Cortex~A76)
static constexpr int H_TILE = 4; // output rows per file 
static constexpr int C_OUT_TILE = 16; // output channels per tile (filter reuse)

void measure_layer_rs(const Convo& layer,
                        const std::string& model){

    auto [in_buff, weights_buff] = data_generating(layer);

    auto rs_results = convo_base(layer.stride, layer.pad);
    rs_results.input.set(in_buff);
    rs_results.weights.set(weights_buff);

    const int W_out_ext = layer.W_out;
    const int H_out_ext = layer.H_out;
    const int C_out_ext = layer.C_out;
    const int tile_H = std::min (H_TILE, H_out_ext); 
    const int tile_C = std::min(C_OUT_TILE, C_out_ext); 
    const int vec = std::min(8, W_out_ext);

    rs_results.convolution.compute_root();

    // Var H_out_outer("H_out_outer"), H_out_inner("H_out_inner");
    // Var C_out_outer("C_out_outer"), C_out_inner("C_out_inner");


    // ??????????????????????????----------------------
    // is the init stage schedule neccesarry?
    // ------------------------------------------------
    // rs_results.convolution
    //     .bound(rs_results.W_out, 0, W_out)
    //     .bound(rs_results.H_out, 0, H_out)
    //     .bound(rs_results.C_out, 0, C_out_ext)
    //     .tile(rs_results.H_out, rs_results.C_out,
    //           H_out_outer, H_out_inner,
    //           C_out_outer, C_out_inner, tile_H, tile_C,
    //         TailStrategy::GuardWithIf)
    //     .reorder(rs_results.W_out, H_out_inner, C_out_inner,
    //              H_out_outer, C_out_outer)
    //     .vectorize(rs_results.W_out, 8)
    //     .parallel(C_out_outer);

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

    rs_results.convolution.update()
                .split(rs_results.H_out, H_out_outer_u, H_out_inner_u, tile_H, TailStrategy::GuardWithIf)
                .split(rs_results.C_out, C_out_outer_u, C_out_inner_u, tile_C, TailStrategy::GuardWithIf)
                .reorder(rs_results.W_out, 
                    H_out_inner_u, 
                    C_out_inner_u,
                    rs_results.r[0], //kW
                    rs_results.r[1], //kH
                    rs_results.r[2], //C_in
                    H_out_outer_u, C_out_outer_u)
                .vectorize(rs_results.W_out, vec, TailStrategy::GuardWithIf)
                .parallel(C_out_outer_u);

    Buffer<float> output(W_out_ext, H_out_ext, C_out_ext);
    PerfResult rs_res = measure_convolution([&](){
        rs_results.convolution.realize(output);
    });

    rs_res.schedule_name = "Row-stationary inspired";
    rs_res.model_name = model;
    rs_res.layer_name = layer.name;
    write_csv(output_path, rs_res);
    std::cout << "[" << layer.name << "] output: "
            << layer.W_out << "x" << layer.H_out << "x" << layer.C_out << "\n";
        
}