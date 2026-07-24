#include <string>
#include "../runner/perf_wrapper.h"
#include <fstream>
#include <iostream>
#include <filesystem>

static std::string current_timestamp()
{
    auto now = std::chrono::system_clock::now();
    std::time_t t = std::chrono::system_clock::to_time_t(now);
    std::ostringstream oss;
    oss << std::put_time(std::localtime(&t), "%Y-%m-%dT%H:%M:%S");
    return oss.str();
}

void write_csv(const std::string &output_path, const PerfResult &layer_result)
{
    std::ofstream file;
    std::filesystem::create_directories(
        std::filesystem::path(output_path).parent_path());

    bool needs_header = !std::filesystem::exists(output_path) || std::filesystem::file_size(output_path) == 0;
    file.open(output_path, std::ofstream::app);
    if (!file.is_open())
    {
        std::cerr << "Failed to open: " << output_path << "\n";
        return;
    }
    if (needs_header)
    {

        file << "timestamp,model,layer,schedule,llc_access,llc_misses,l2_refill,l2_read,l2_write\n";
    }

    file << current_timestamp() << ","
         << layer_result.model_name << ","
         << layer_result.layer_name << ","
         << layer_result.schedule_name << ","
         << layer_result.llc_accesses << ","
         << layer_result.llc_misses << ","
         << layer_result.l2_refill << ","
         << layer_result.l2_read << ","
         << layer_result.l2_write << "\n"
         << layer_result.l2_write_back << "\n";

    file.close();
};