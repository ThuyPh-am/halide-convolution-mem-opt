#include <stdio.h>
#include <vector>
#include <string>
#include <Halide.h>
#include "../config/layer_definitions.h"
#include "convolution_base.h"
#include "../runner/data_generator.h"
#include "../runner/perf_wrapper.h"
#include "../utils/csv_writer.h"

using namespace Halide;

std::string output_path = "../results/raw/raw_results.csv";

void measure_layer_ws(const Convo &layer,
                    const std::string &model)
{

    // prep input, weights and pipeline set up
    Buffer<float> output = calculate_output(layer);
    auto [in_buff, weights_buff] = data_generating(layer);

    auto ws_result = convo_base(layer.stride, layer.pad); // auto = {convolution, input, weights}
    ws_result.input.set(in_buff);
    ws_result.weights.set(weights_buff);

    ws_result.convolution.update()
                        .reorder(ws_result.W_out, 
                            ws_result.H_out, ws_result.r[2], 
                            ws_result.r[0], ws_result.r[1], ws_result.C_out);

    ws_result.convolution.compute_root();
    // counting and realize
    PerfResult r = measure_convolution([&](){
        ws_result.convolution.realize(output);
    });

    // writing csv
    r.model_name = model;
    r.schedule_name = "weight_stationary_inspired";
    r.layer_name = layer.name;
    write_csv(output_path, r);
}