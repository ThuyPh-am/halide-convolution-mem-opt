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

std::string output_path = "../results/raw/raw_results.csv";

void measure_layer_os(const Convo& layer,
                    const std::string& model){
    // outside counter: data prep
    Buffer<float> output = calculate_output(layer);
    auto [in_buff, weights_buff] = data_generating(layer);

    // outside counter: pipeline set up
    auto os_result = convo_base(layer.stride, layer.pad);
    os_result.input.set(in_buff);
    os_result.weights.set(weights_buff);

    // parallel C_out
    // outer tile H -> outer tile W
    // inner tile H (8) -> inner tile W (8, vectorized)
   int tile_w = std::min(8, layer.W_out);
   int tile_h = std::min(8, layer.H_out);
    Var W_out_outer ("W_out_outer"), W_out_inner("W_out_inner"), 
        H_out_outer ("H_out_outer"), H_out_inner ("H_out_inner");
    Var tile_index("tile_index");
    os_result.convolution
                    .tile(os_result.W_out, os_result.H_out, 
                    W_out_outer, H_out_outer,
                    W_out_inner, H_out_inner, tile_w,tile_h)
                    .vectorize(W_out_inner)
                    .fuse(W_out_outer, H_out_outer, tile_index)
                    .parallel(tile_index);
    // r[kW] -> r[kH] -> r[C_in]

    os_result.convolution.update()
                        .reorder(os_result.r[0], os_result.r[1], os_result.r[2],
                                W_out_inner, H_out_inner, W_out_outer, H_out_outer,
                                os_result.C_out);

    os_result.convolution.compute_root();
    // run counter
    PerfResult os_res = measure_convolution([&](){
        os_result.convolution.realize(output);
    });
    // write csv
    os_res.model_name = model;
    os_res.schedule_name = "os_inspired";
    os_res.layer_name = layer.name;
    write_csv(output_path, os_res);

};
