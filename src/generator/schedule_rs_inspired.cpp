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

// approximation of RS on CPU:
// 1. A tile of output rows stays in L1/L2 while i iterate over
    // every filter row and every input channel that feeds it -> input-row reuse
// 2. The reduction is ordered r[2] (C_in) -> r[1] (kH) -> r[0] [kW]
    // so the innermost access pattern matches RS: for a fixed filter row,
    // sweep the input row spatially (kW), then advance to the next filter row
// 3. Output tiles are vectorised over W_out to exploit SIMD and keep prtial
    // sum in registers during the ineermost loop
std::string output_path = "../results/raw/raw_results.csv";

// tile size------------------------------------------
// Choosen to fit a tile of (H_tile output rows) x (W_out) partial sums plus
// the corresponding input row into L2 (~512 KB on Cortex~A76)
// Adjust if profilling shows thrashing
static constexpr int H_TILE = 4; // output rows per file 
static constexpr int C_OUT_TILE = 16; // output channels per tile (filter reuse)

void measure_layer_rs(const Convo& layer,
                        const std::string& model){

    Buffer<float> output = calculate_output(layer);
    auto [in_buff, weights_buff] = data_generating(layer);

    auto rs_results = convo_base(layer.stride, layer.pad);
    rs_results.input.set(in_buff);
    rs_results.weights.set(weights_buff);

    int tile_H = std::min (H_TILE, layer.H_out);
    int tile_C = std::min(C_OUT_TILE, layer.C_out);

    Var H_out_outer("H_out_outer"), H_out_inner("H_out_inner");
    Var C_out_outer("C_out_outer"), C_out_inner("C_out_inner");

    rs_results.convolution.compute_root();

    // ??????????????????????????----------------------
    // is the init stage schedule neccesarry?
    // ------------------------------------------------
    rs_results.convolution
        .tile(rs_results.H_out, rs_results.C_out,
              H_out_outer, H_out_inner,
              C_out_outer, C_out_inner, tile_H, tile_C)
        .reorder(rs_results.W_out, H_out_inner, C_out_inner,
                 H_out_outer, C_out_outer)
        .vectorize(rs_results.W_out, 8)
        .parallel(C_out_outer);

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

    rs_results.convolution.update()
                .reorder(rs_results.W_out, 
                    H_out_inner, 
                    C_out_inner,
                    rs_results.r[0],
                    rs_results.r[1],
                    rs_results.r[2],
                    H_out_outer, C_out_outer)
                .vectorize(rs_results.W_out,8)
                .parallel(C_out_outer);

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