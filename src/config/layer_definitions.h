// Define header file contains layer definitions of AlexNet, VGG16, MobileNet

// AlexNet
// Eyeriss: A Spatial Architecture for Energy-Efficient Dataflow for Convolutional Neural Networks
// Table II, p4

// N: batch size of 3D fmaps
// C_in: # of ifmap/ filter channels
// C_out: # of 3D filters/ # of ofmap channels
// H_in: ifmap plane height
// W_in: ifmap plane width
// K_h: filter plane height
// K_w: filter plane width
// H_out: ofmap plane height
// W_out: ofmap plane width

#pragma once
#include <string>
#include <vector>

constexpr int out_dim(int in, int pad, int k, int stride)
{
    return (in + 2 * pad - k) / stride + 1;
}

struct Convo
{
    int C_in, H_in, W_in, C_out;
    int K_h, K_w;
    int H_out, W_out;
    int stride, pad;
    std::string name;

    Convo(int C_in, int H_in, int W_in, int C_out,
          int K_h, int K_w, int stride, int pad, std::string name)
        : C_in(C_in), H_in(H_in), W_in(W_in), C_out(C_out),
          K_h(K_h), K_w(K_w), stride(stride), pad(pad),
          H_out(out_dim(H_in, pad, K_h, stride)),
          W_out(out_dim(W_in, pad, K_w, stride)),
          name(std::move(name)) {}
};

inline const Convo Test_layer(4, 8, 8, 8, 3, 3, 1, 1, "tiny_3x3_4to8");

namespace AlexNet
{
    inline const Convo conv1(3, 227, 227, 96, 11, 11, 4, 0, "Alex_conv1");
    inline const Convo conv2(48, 31, 31, 256, 5, 5, 1, 0, "Alex_conv2");
    inline const Convo conv3(256, 15, 15, 384, 3, 3, 1, 0, "Alex_conv3");
    inline const Convo conv4(192, 15, 15, 384, 3, 3, 1, 0, "Alex_conv4");
    inline const Convo conv5(192, 15, 15, 256, 3, 3, 1, 0, "Alex_conv5");
    inline const std::vector<Convo> layers = {conv1, conv2, conv3, conv4, conv5};
}

namespace VGG16
{
    inline const Convo conv1_1(3, 224, 224, 64, 3, 3, 1, 1, "VGG_conv1_1");
    inline const Convo conv1_2(64, 224, 224, 64, 3, 3, 1, 1, "VGG_conv1_2");
    inline const Convo conv2_1(64, 112, 112, 128, 3, 3, 1, 1, "VGG_conv2_1");
    inline const Convo conv3_1(128, 56, 56, 256, 3, 3, 1, 1, "VGG_conv3_1");
    inline const Convo conv4_1(256, 28, 28, 512, 3, 3, 1, 1, "VGG_conv4_1");
    inline const std::vector<Convo> layers = {conv1_1, conv1_2, conv2_1, conv3_1, conv4_1};
}

namespace MobileNet
{
    inline const Convo conv1(3, 224, 224, 32, 3, 3, 1, 1, "MobileNet_conv1");
    inline const Convo dw2(32, 112, 112, 32, 3, 3, 1, 1, "MobileNet_dw2");
    inline const Convo pw2(32, 112, 112, 64, 1, 1, 1, 0, "MobileNet_pw2");
    inline const Convo dw3(64, 56, 56, 64, 3, 3, 1, 1, "MobileNet_dw3");
    inline const std::vector<Convo> layers = {conv1, dw2, pw2, dw3};
}

struct ModelDef
{
    std::string name;
    std::vector<Convo> layers;
};

const std::vector<ModelDef> ALL_MODELS = {
    {"AlexNet", AlexNet::layers},
    {"VGG16", VGG16::layers},
    {"MobileNet", MobileNet::layers},
};