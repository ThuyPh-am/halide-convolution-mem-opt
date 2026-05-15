#include <stdio.h>
#include <vector>
#include <iostream>
#include <algorithm>
#include <Halide.h>
#include "../config/layer_definitions.h"
#include "convolution_base.h"
#include "../runner/perf_wrapper.h"
#include "../runner/data_generator.h"
#include "../utils/csv_writer.h"

using namespace Halide;

static constexpr bool INSPECT = true; 

static const std::string output_path = "../results/raw/raw_results.csv";

static constexpr int H_TILE = 8;      // output rows per file
static constexpr int W_TILE = 8; // ou
void measure_layer_os(const Convo& layer,
                    const std::string& model){

    // outside counter: data prep
    auto [in_buff, weights_buff] = data_generating(layer);

    // outside counter: pipeline set up
    auto os_result = convo_base(layer.stride, layer.pad);
    os_result.input.set(in_buff);
    os_result.weights.set(weights_buff);

    // parallel C_out
    // outer tile H -> outer tile W
    // inner tile H (8) -> inner tile W (8, vectorized)
    os_result.convolution.compute_root();

    const int W_out_ext = layer.W_out;
    const int H_out_ext = layer.H_out;
    const int C_out_ext = layer.C_out;
    const int tile_w = std::min(W_TILE, W_out_ext);
    const int tile_h = std::min(H_TILE, H_out_ext);
    Var W_out_outer ("W_out_outer"), W_out_inner("W_out_inner"), 
        H_out_outer ("H_out_outer"), H_out_inner ("H_out_inner");
    Var tile_index("tile_index");
    os_result.convolution
        .bound(os_result.W_out, 0, W_out_ext)
        .bound(os_result.H_out, 0, H_out_ext)
        .bound(os_result.C_out, 0, C_out_ext)
        .tile(os_result.W_out, os_result.H_out,
              W_out_outer, H_out_outer,
              W_out_inner, H_out_inner, tile_w, tile_h,
              Halide::TailStrategy::GuardWithIf)
        .vectorize(W_out_inner)
        .fuse(W_out_outer, H_out_outer, tile_index)
        .parallel(tile_index);
    // r[kW] -> r[kH] -> r[C_in]

    os_result.convolution.update()
        .tile(os_result.W_out, os_result.H_out,
              W_out_outer, H_out_outer,
              W_out_inner, H_out_inner, tile_w, tile_h,
            Halide::TailStrategy::GuardWithIf)
        .reorder(os_result.r[0], os_result.r[1], os_result.r[2],
                 W_out_inner, H_out_inner, W_out_outer, H_out_outer,
                 os_result.C_out)
        .fuse(W_out_outer, H_out_outer, tile_index)
        .parallel(tile_index)
        .vectorize(W_out_inner);

    if (INSPECT)
    {
        std::cout << "\n======OS loop nest for [" << layer.name << "]====\n";
        os_result.convolution.print_loop_nest();
        std::cout << "======= end ======= \n\n";
    }

    // run counter
    Buffer<float> output(W_out_ext, H_out_ext, C_out_ext);
    PerfResult os_res = measure_convolution([&](){
        os_result.convolution.realize(output);
    });
    // write csv
    os_res.model_name = model;
    os_res.schedule_name = "output_stationary_inspired";
    os_res.layer_name = layer.name;
    write_csv(output_path, os_res);

};
