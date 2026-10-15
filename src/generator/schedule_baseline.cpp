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

static constexpr bool INSPECT = true;

static const std::string output_path = "../results/raw/rerun_results.csv";

void measure_layer_baseline(const Convo &layer,
                            const std::string &model)
{
    // outside counter: data prep
    auto [in_buff, weights_buff] = data_generating(layer);

    // outisde counter: pipeline setup
    auto result = convo_base(layer.stride, layer.pad); // result = [convolution, input, weights, r, W_out, H_out, C_out]

    result.input.set(in_buff);
    result.weights.set(weights_buff);

    result.convolution.compute_root();
    // start counter only before realize
    if (INSPECT)
    {
        std::cout << "\n======Baseline loop nest for [" << layer.name << "]====\n";
        result.convolution.print_loop_nest();
        std::cout << "============== \n\n";
    }
    result.convolution.compile_jit(); // compile outside the counted window
    Buffer<float> output(layer.W_out, layer.H_out, layer.C_out);
    output.fill(0.0f); // fault output pages in before counting
    PerfResult r = measure_convolution([&]()
                                       { result.convolution.realize(output); });
    r.model_name = model;
    r.schedule_name = "baseline";
    r.layer_name = layer.name;
    write_csv(output_path, r);
    std::cout << "[" << layer.name << "] output: "
              << layer.W_out << "x" << layer.H_out << "x" << layer.C_out << "\n";
    std::cout << "======= end ======= \n\n";
}