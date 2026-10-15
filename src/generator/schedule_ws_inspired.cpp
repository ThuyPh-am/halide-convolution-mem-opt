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

static constexpr bool INSPECT = true;

static const std::string output_path = "../results/raw/rerun_results.csv";

void measure_layer_ws(const Convo &layer,
                      const std::string &model)
{

    // prep input, weights and pipeline set up
    auto [in_buff, weights_buff] = data_generating(layer);

    auto ws_result = convo_base(layer.stride, layer.pad); // auto = {convolution, input, weights}
    ws_result.input.set(in_buff);
    ws_result.weights.set(weights_buff);

    ws_result.convolution.compute_root();

    ws_result.convolution.update()
        .reorder(ws_result.W_out,
                 ws_result.H_out, ws_result.r[2],
                 ws_result.r[0], ws_result.r[1], ws_result.C_out);

    if (INSPECT)
    {
        std::cout << "\n======WS loop nest for [" << layer.name << "]====\n";
        ws_result.convolution.print_loop_nest();
        std::cout << "============== \n\n";
    }

    // counting and realize
    ws_result.convolution.compile_jit(); // compile outside the counted window
    Buffer<float> output(layer.W_out, layer.H_out, layer.C_out);
    output.fill(0.0f); // fault output pages in before counting
    PerfResult r = measure_convolution([&]()
                                       { ws_result.convolution.realize(output); });

    // writing csv
    r.model_name = model;
    r.schedule_name = "weight_stationary_inspired";
    r.layer_name = layer.name;
    write_csv(output_path, r);
    std::cout << "[" << layer.name << "] output: "
              << layer.W_out << "x" << layer.H_out << "x" << layer.C_out << "\n";
    std::cout << "======= end ======= \n\n";
}