
// approximation of RS on CPU:
// 1. A tile of output rows stays in L1/L2 while i iterate over
    // every filter row and every input channel that feeds it -> input-row reuse
// 2. The reduction is ordered r[2] (C_in) -> r[1] (kH) -> r[0] [kW]
    // so the innermost access pattern matches RS: for a fixed filter row,
    // sweep the input row spatially (kW), then advance to the next filter row
// 3. Output tiles are vectorised over W_out to exploit SIMD and keep prtial
    // sum in registers during the ineermost loop