#include <halide.h>
#include <algorithm>
#include <stdio.h>
#include <../generator/halide_loop_inspect.h>

using namespace Halide;

static constexpr int H_TILE = 4;
static constexpr int C_OUT_TILE = 16;


int apply_rs(const Convo& layer, const std::string& model)
{

}

int main(){

    Convo alex_conv1 = {"Alex_conv1", 3, 227, 227, 96, 11, 11, 4, 0, 55, 55};
    Convo vgg_conv1_2 = {"VGG_conv1_2", 64, 224, 224, 64, 3, 3, 1, 1, 224, 224};

    inspect_schedule(alex_conv1, "RS", apply_rs);
    inspect_schedule(vgg_conv1_2, "RS", apply_rs);
}