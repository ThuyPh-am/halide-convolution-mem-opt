#include <string>
#include "../runner/perf_wrapper.h"
#include <fstream>
#include <iostream>
#include <filesystem>

static std::string current_timestamp(){
    auto now = std::chrono::system_clock::now();
    std::time_t t = std::chrono::system_clock::to_time_t(now);
    std::ostringstream oss;
    oss << std::put_time(std::localtime(&t), "%Y-%m-%dT%H:%M:%S");
    return oss.str();
}

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

    file << current_timestamp() << "," 
        <<layer_result.model_name << ","
        <<layer_result.layer_name << ","
        <<layer_result.schedule_name << ","
        <<layer_result.llc_accesses <<"," 
        <<layer_result.llc_misses<<"\n";

    file.close();
};