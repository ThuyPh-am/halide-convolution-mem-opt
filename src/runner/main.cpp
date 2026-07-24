#include "../config/layer_definitions.h"
#include "../generator/schedule_baseline.h"
#include "../generator/schedule_os_inspired.h"
#include "../generator/schedule_rs_inspired.h"
#include "../generator/schedule_ws_inspired.h"
#include <iostream>
#include <Halide.h>
#include <vector>

void flush_cache()
{
    // Allocate and touch a buffer larger than L3
    // to evict all cached data
    static constexpr size_t FLUSH_SIZE = 4 * 1024 * 1024; // 4MB > 2MB L3
    std::vector<char> flush_buf(FLUSH_SIZE, 0);
    for (size_t i = 0; i < FLUSH_SIZE; i += 64)
    {
        flush_buf[i] = i & 0xFF;
    }
    // prevent compiler from optimizing away
    volatile char x = flush_buf[FLUSH_SIZE - 1];
    (void)x;
}

int main()
{
    setenv("HL_DEBUG_CODEGEN", "0", 1);
    setenv("HL_NO_ASSERTS", "0", 1);

    try
    {
        for (const auto &model : ALL_MODELS)
        {
            for (const auto &layer : model.layers)
            {
                flush_cache();
                std::cout << "[running] " << model.name << " " << layer.name << " baseline\n";
                measure_layer_baseline(layer, model.name);
                flush_cache();
                std::cout << "[running] " << model.name << " " << layer.name << " rs\n";
                measure_layer_rs(layer, model.name);
                flush_cache();
                std::cout << "[running] " << model.name << " " << layer.name << " ws\n";
                measure_layer_ws(layer, model.name);
                flush_cache();
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