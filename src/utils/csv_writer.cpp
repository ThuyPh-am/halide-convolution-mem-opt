#include <string>
#include "../runner/perf_wrapper.h"
#include <fstream>
#include <iostream>
#include <filesystem>


void write_csv(const std::string& output_path, const PerfResult& layer_result){
    std::ofstream file;
    std::filesystem::create_directories(
        std::filesystem::path(output_path).parent_path()
    );
    file.open(output_path, std::ofstream::app);
    if (!file.is_open())
    {
        std::cerr << "Failed to open: " << output_path << "\n";
        return; 
    }

    bool needs_header = !std::filesystem::exists(output_path) || std::filesystem::file_size(output_path) == 0;
    if (needs_header)
    {
        file << "model,layer,schedule,llc_misses\n";
    }

    file << layer_result.model_name << ","
        <<layer_result.layer_name << ","
        <<layer_result.schedule_name << ","
        <<layer_result.llc_misses << "\n";

    file.close();
};