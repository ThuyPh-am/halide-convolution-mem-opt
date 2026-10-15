#include "../config/layer_definitions.h"
#include "../generator/schedule_baseline.h"
#include "../generator/schedule_os_inspired.h"
#include "../generator/schedule_rs_inspired.h"
#include "../generator/schedule_ws_inspired.h"
#include <iostream>
#include <Halide.h>
#include <vector>

int main()
{
    setenv("HL_DEBUG_CODEGEN", "0", 1);
    setenv("HL_NO_ASSERTS", "0", 1);

    try
    {
        // floor check: an 8x8 layer needs almost no DRAM traffic
        measure_layer_baseline(Test_layer, "Test");
        static constexpr int REPS = 5;
        for (int rep = 0; rep < REPS; ++rep)
        for (const auto &model : ALL_MODELS)
        {
            for (const auto &layer : model.layers)
            {
                std::cout << "[running] " << model.name << " " << layer.name << " baseline\n";
                measure_layer_baseline(layer, model.name);
                std::cout << "[running] " << model.name << " " << layer.name << " rs\n";
                measure_layer_rs(layer, model.name);
                std::cout << "[running] " << model.name << " " << layer.name << " ws\n";
                measure_layer_ws(layer, model.name);
                std::cout << "[running] " << model.name << " " << layer.name << " os\n";
                measure_layer_os(layer, model.name);
            }
        }
    }
    catch (const Halide::CompileError &e)
    {
        std::cerr << "Halide::CompileError: " << e.what() << "\n";
        return 1;
    }
    catch (const Halide::RuntimeError &e)
    {
        std::cerr << "Halide::RunTimeError: " << e.what() << "\n";
        return 1;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception: " << e.what() << "\n";
        return 1;
    }
    return 0;
}