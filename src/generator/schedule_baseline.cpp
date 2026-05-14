#include <stdio.h>
#include <vector>
#include <iostream>
#include <Halide.h>
#include "../config/layer_definitions.h"
#include "convolution_base.h"
#include "../runner/perf_wrapper.h"
#include "../runner/data_generator.h"
#include "../utils/csv_writer.h"

// struct Convo
// {
//     // int N; // batch size will be pass separately in runtime
//     int C_in;  // input channels
//     int H_in;  // input height
//     int W_in;  // input width
//     int C_out; // output channels
//     int K_h;   // kernel height
//     int K_w;   // kernel width
//     int H_out; // output height
//     int W_out; // output width
//     int stride;
//     int pad;
//     std::string name;
// };

using namespace Halide;

static const std::string output_path = "../results/raw/raw_results.csv";

void measure_layer_baseline(const Convo& layer,
                const std::string& model){
    // outside counter: data prep
    auto [in_buff, weights_buff] = data_generating(layer);

    // outisde counter: pipeline setup
    auto result = convo_base(layer.stride, layer.pad); // result = [convolution, input, weights]

    result.input.set(in_buff);
    result.weights.set(weights_buff);

    result.convolution.compute_root();
    // start counter only before realize
    Buffer<float> output(layer.W_out, layer.H_out, layer.C_out);
    PerfResult r = measure_convolution([&](){
        result.convolution.realize(output);
    });
    r.model_name = model;
    r.schedule_name = "baseline";
    r.layer_name = layer.name;
    write_csv(output_path, r);
}