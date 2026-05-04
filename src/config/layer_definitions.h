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

struct Convo
{
    // int N; // batch size will be pass separately in runtime
    int C_in; // input channels
    int H_in; //input height
    int W_in; // input width
    int C_out; //output channels
    int K_h; // kernel height
    int K_w; // kernel width
    int H_out; //output height
    int W_out; // output width
    int stride;
    int pad;
    std::string name;
};

constexpr int out_dim(int in, int pad, int k, int stride){
    return (in + 2 * pad - k) /stride + 1;
};
inline const struct Convo Test_layer = {
    .C_in = 4,
    .H_in = 8,
    .W_in = 8,
    .C_out = 8,
    .K_h = 3,
    .K_w = 3,
    .stride = 1,
    .pad = 1,
    .name = "tiny_3x3_4to8"
};

namespace AlexNet
{
    // batch size 4 for eyeriss (p. 9 Eyeriss: An Energy-Efficient Reconfigurable 
    // Accelerator for Deep Convolutional Neural Networks)
    const int conv_num = 5;
    const int fc_num = 3;
    inline const Convo conv1 = {
        .C_in = 3,
        .H_in = 227,
        .W_in = 227,
        .C_out = 96,
        .K_h = 11,
        .K_w = 11,
        .stride = 4,
        .pad = 1,
        .name = "Alex_conv1"};
    inline const Convo conv2 = {
        .C_in = 48,
        .H_in = 31,
        .W_in = 31,
        .C_out = 256,
        .K_h = 5,
        .K_w = 5,
        .stride = 1,
        .pad = 1,
        .name = "Alex_conv2"};
    inline const Convo conv3 = {       
        .C_in = 256,
        .H_in = 15,
        .W_in = 15,
        .C_out = 384,
        .K_h = 3,
        .K_w = 3,
        .stride = 1,
        .pad = 1,
        .name = "Alex_conv3"};
    inline const Convo conv4 = {
        .C_in = 192,
        .H_in = 15,
        .W_in = 15,
        .C_out = 384,
        .K_h = 3,
        .K_w = 3,
        .stride = 1,
        .pad = 1,
        .name = "Alex_conv4"
    };
    inline const Convo conv5 = {        
        .C_in = 192,
        .H_in = 15,
        .W_in = 15,
        .C_out = 256,
        .K_h = 3,
        .K_w = 3,
        .stride = 1,
        .pad = 1,
        .name = "Alex_conv5"};

    inline const std::vector<Convo> layers = {conv1, conv2, conv3, conv4, conv5};
};

namespace VGG16
{
    const int conv_num = 5;
    // batch size 3 for eyeriss (p. 9 Eyeriss: An Energy-Efficient Reconfigurable 
    // Accelerator for Deep Convolutional Neural Networks)
    inline const Convo conv1_1 = {        
        .C_in = 3,
        .H_in = 224,
        .W_in = 224,
        .C_out = 64,
        .K_h = 3,
        .K_w = 3,
        .stride = 1,
        .pad = 1,
        .name = "VGG_conv1_1"};
    inline const Convo conv1_2 = {        
        .C_in = 64,
        .H_in = 224,
        .W_in = 224,
        .C_out = 64,
        .K_h = 3,
        .K_w = 3,
        .stride = 1,
        .pad = 1,
        .name = "VGG_conv1_2"};
    inline const Convo conv2_1 = {        
        .C_in = 64,
        .H_in = 112,
        .W_in = 112,
        .C_out = 128,
        .K_h = 3,
        .K_w = 3,
        .stride = 1,
        .pad = 1,
        .name = "VGG_conv2_1"};
    inline const Convo conv3_1 = {        
        .C_in = 128,
        .H_in = 56,
        .W_in = 56,
        .C_out = 256,
        .K_h = 3,
        .K_w = 3,
        .stride = 1,
        .pad = 1,
        .name = "VGG_conv3_1"};
    inline const Convo conv4_1 = {        
        .C_in = 256,
        .H_in = 28,
        .W_in = 28,
        .C_out = 512,
        .K_h = 3,
        .K_w = 3,
        .stride = 1,
        .pad = 1,
        .name = "VGG_conv4_1"};

    inline const std::vector<Convo> layers ={conv1_1, conv1_2, conv2_1, conv3_1, conv4_1};
};

namespace MobileNet
{
    const int conv_num = 4;
    inline const Convo conv1 = {        
        .C_in = 3,
        .H_in = 224,
        .W_in = 224,
        .C_out = 32,
        .K_h = 3,
        .K_w = 3,
        .stride = 1,
        .pad = 1,
        .name = "Mobilenet_conv1"};
    inline const Convo dw2 = {        
        .C_in = 32,
        .H_in = 112,
        .W_in = 112,
        .C_out = 32,
        .K_h = 3,
        .K_w = 3,
        .stride = 1,
        .pad = 1,
        .name = "MobileNet_dw2"};
    inline const Convo pw2 = {        
        .C_in = 32,
        .H_in = 112,
        .W_in = 112,
        .C_out = 64,
        .K_h = 1,
        .K_w = 1,
        .stride = 1,
        .pad = 0,
        .name = "MobileNet_pw2"};
    inline const Convo dw3 = {        
        .C_in = 64,
        .H_in = 56,
        .W_in = 56,
        .C_out = 64,
        .K_h = 3,
        .K_w = 3,
        .stride = 1,
        .pad = 1,
        .name = "MobileNet_dw3"};

    inline const std::vector<Convo> layers = {conv1, dw2, pw2, dw3};
};

struct ModelDef {
    std::string name;
    std::vector<Convo> layers;
};

const std::vector<ModelDef> ALL_MODELS = {
    {"AlexNet", AlexNet::layers},
    {"VGG16", VGG16::layers},
    {"MobileNet", MobileNet::layers},
};