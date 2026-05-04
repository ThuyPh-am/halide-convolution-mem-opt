// convolution base equation

#include <iostream>
#include <stdio.h>
#include <Halide.h>
#include "convolution_base.h"

using namespace Halide;

ConvResult convo_base(int stride, int pad){
    // // === Intputs ===
    ImageParam input(Float(32), 3, "input"); // W, H, C_in
    ImageParam weights(Float(32), 4, "weights"); // kW, kH, C_in, C_out

    //PAD THE INPUT WITH ZEROS
    Func padded("padded");
    padded = BoundaryConditions::constant_exterior(input, 0.0f);

    // === Outputs === 
    Var W_out("W_out"), H_out("H_out"), C_out("C_out");

    //  === Reduction domain: filter width, filter height, input channels ===
    RDom r(0, weights.dim(0).extent(), // kw
            0, weights.dim(1).extent(), // kh
            0, weights.dim(2).extent(), // Cin
            "r");

    // === Convolution function ===
    Func convolution("convolution");
    convolution(W_out, H_out, C_out) = 0.0f;
    convolution(W_out, H_out, C_out) += 
                                    padded(W_out * stride + r[0] - pad, H_out * stride + r[1] - pad, r[2]) * 
                                    weights(r[0], r[1], r[2], C_out);

    return {convolution, input, weights, r, W_out, H_out, C_out};
}
    