// Correctness check of the four schedules (directives copied from the
// schedule_*.cpp files) against a plain C++ reference convolution, on random data.
#include <Halide.h>
#include <cstdio>
#include <cmath>
#include <random>
#include <algorithm>
#include <string>
#include "convolution_base.h"
using namespace Halide;

struct L { int Cin, H, W, Cout, K, stride, pad; const char* name; };
static int od(int in, int p, int k, int s) { return (in + 2*p - k)/s + 1; }

static void sched(ConvResult& c, const std::string& s, const L& l, int Ho, int Wo, bool print) {
    c.convolution.compute_root();
    if (s == "rs") {
        Var ho("ho"), hi("hi"), co("co"), ci("ci");
        c.convolution.update()
          .split(c.H_out, ho, hi, std::min(4, Ho), TailStrategy::GuardWithIf)
          .split(c.C_out, co, ci, std::min(4, l.Cout), TailStrategy::GuardWithIf)
          .reorder(c.W_out, hi, ci, c.r[0], c.r[1], c.r[2], co, ho)
          .vectorize(c.W_out, std::min(8, Wo), TailStrategy::GuardWithIf);
    } else if (s == "ws") {
        c.convolution.update().reorder(c.W_out, c.H_out, c.r[2], c.r[0], c.r[1], c.C_out);
    } else if (s == "os") {
        Var wo("wo"), wi("wi"), ho("ho"), hi("hi"), t("t");
        int tw = std::min(8, Wo), th = std::min(8, Ho);
        c.convolution.bound(c.W_out, 0, Wo).bound(c.H_out, 0, Ho).bound(c.C_out, 0, l.Cout)
          .tile(c.W_out, c.H_out, wo, ho, wi, hi, tw, th, TailStrategy::GuardWithIf)
          .vectorize(wi).fuse(wo, ho, t);
        c.convolution.update()
          .tile(c.W_out, c.H_out, wo, ho, wi, hi, tw, th, TailStrategy::GuardWithIf)
          .reorder(c.r[0], c.r[1], c.r[2], wi, hi, wo, ho, c.C_out)
          .fuse(wo, ho, t).vectorize(wi);
    }
    if (print) { printf("--- %s loop nest (%s)\n", s.c_str(), l.name); c.convolution.print_loop_nest(); }
}

int main() {
    std::mt19937 g(1); std::uniform_real_distribution<float> U(-1, 1);
    L layers[] = {
        {3, 227, 227, 8, 11, 4, 0, "alex_conv1-like"},
        {6, 31, 31, 10, 5, 1, 0, "alex_conv2-like"},
        {5, 15, 15, 7, 3, 1, 0, "alex_conv3-like"},
        {4, 30, 29, 6, 3, 1, 1, "vgg-like odd tails"},
        {5, 20, 20, 9, 1, 1, 0, "pointwise 1x1"},
    };
    bool all_ok = true; bool printed = false;
    for (auto& l : layers) {
        int Ho = od(l.H, l.pad, l.K, l.stride), Wo = od(l.W, l.pad, l.K, l.stride);
        Buffer<float> in(l.W, l.H, l.Cin), w(l.K, l.K, l.Cin, l.Cout);
        in.for_each_value([&](float& v){ v = U(g); });
        w.for_each_value([&](float& v){ v = U(g); });
        Buffer<float> ref(Wo, Ho, l.Cout);
        for (int co = 0; co < l.Cout; co++) for (int y = 0; y < Ho; y++) for (int x = 0; x < Wo; x++) {
            double s = 0;
            for (int ci = 0; ci < l.Cin; ci++) for (int ky = 0; ky < l.K; ky++) for (int kx = 0; kx < l.K; kx++) {
                int iy = y*l.stride + ky - l.pad, ix = x*l.stride + kx - l.pad;
                if (iy >= 0 && iy < l.H && ix >= 0 && ix < l.W) s += in(ix, iy, ci) * w(kx, ky, ci, co);
            }
            ref(x, y, co) = (float)s;
        }
        for (std::string s : {"baseline", "rs", "ws", "os"}) {
            auto c = convo_base(l.stride, l.pad);
            c.input.set(in); c.weights.set(w);
            sched(c, s, l, Ho, Wo, !printed && l.stride == 1 && l.pad == 1);
            Buffer<float> out(Wo, Ho, l.Cout);
            c.convolution.realize(out);
            float md = 0;
            out.for_each_element([&](int x, int y, int z){ md = std::max(md, std::fabs(out(x,y,z) - ref(x,y,z))); });
            bool ok = md < 1e-3f; all_ok &= ok;
            printf("%-20s %-9s out %dx%dx%d  max|diff| = %.2e  %s\n", l.name, s.c_str(), Wo, Ho, l.Cout, md, ok ? "OK" : "MISMATCH");
        }
        if (l.stride == 1 && l.pad == 1) printed = true;
    }
    printf(all_ok ? "ALL CORRECT\n" : "SOME MISMATCH\n");
    return all_ok ? 0 : 1;
}
