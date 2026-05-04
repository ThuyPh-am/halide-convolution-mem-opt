 #include <iostream>
 #include <Halide.h>
 #include "convolution_base.h"

 using namespace Halide;

 int main(){
    ConvResult res = convo_base(1);

    // === Realize on a small test case ===
    // 4x4 input, 1 channel, 2x2 kernel, 1 output filter
    Buffer<float> in_buf(4,4,1);
    Buffer<float> w_buf(2,2,1,1);

    float in_data[4][4] = {
            {1,2,3,4},
            {4,5,6,7},
            {2,2,3,4},
            {1,4,5,7},};

    for (int h = 0; h < 4; h++) {
        for (int w = 0; w < 4; w++){
            in_buf(w,h,0) = in_data[h][w];
        }
    }

    // kernel [[1,2],[0,5]]
    w_buf(0,0,0,0) = 1; w_buf(1,0,0,0) = 2;
    w_buf(0,1,0,0) = 0; w_buf(1,1,0,0) = 5;

    res.input.set(in_buf);
    res.weights.set(w_buf);

    // output is 3x3 (4-2+1) for stride = 1, no padding
    Buffer<float> output = res.convolution.realize({3,3,1});

    std::cout<<"Output (3x3):" << std::endl;
    for (int h = 0; h < 3; h++){
        for (int w = 0; w < 3; w++){
            std::cout<<output(w,h,0) << "\t";
        }
        std::cout<<std::endl;
    }
 }
  