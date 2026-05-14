// Wraps a kernel execution with Linux perf_event_open syscall to measure
// LLC-load-misses (proxy for DRAM accesses) and wall-clock time.
// Design to isolated convolution counter
#include "perf_wrapper.h"
#include <linux/perf_event.h>
#include <stdint.h>
#include <sys/ioctl.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <iostream>
#include <functional>
#include <cstring>
#include <cerrno>

// ARM Cortex A76 PMU event numbers (ARMv8 architecture)
// Source: ARM Architecture Reference Manual / Cortex A76 TRM
static constexpr uint32_t ARM_PMU_L3D_CACHE_REFILL = 0x002A; // l3d_cache_refill
static constexpr uint32_t ARM_PMU_L3D_CACHE = 0x002B; // l3d_cache_rd (total)

static int open_raw_pmu_event(uint32_t event_id){
    struct perf_event_attr attr;
    memset(&attr, 0, sizeof(attr));
    attr.type = PERF_TYPE_RAW;
    attr.size = sizeof(attr);
    attr.config = event_id;
    attr.disabled = 1;
    attr.exclude_kernel = 0; // include kernel - l3 refills can be kernel-inititated
    attr.exclude_hv = 1;

    int fd = syscall(SYS_perf_event_open, &attr, 
                0, // current process
                -1, // any CPU
                -1, //no group
                0 );
    if (fd == -1){
        std::cerr << "[perf] perf_event_open failed for event 0x"
                << std::hex << event_id << std::dec
                << ": "<< strerror(errno) << "\n";
    }

    return fd; // -1 on failure
}

PerfResult measure_convolution(std::function<void()>conv_fn){
    PerfResult result;
    result.llc_misses = -1; // measurement unavailable

    int fd_refill = open_raw_pmu_event(ARM_PMU_L3D_CACHE_REFILL);
    int fd_access = open_raw_pmu_event(ARM_PMU_L3D_CACHE);

    // rum the kernel even if counters are unavailable
    if (fd_refill == -1 || fd_access == -1){
        std::cerr << "[perf] Falling back to mock mode (no LLC counts). \n"
                <<"          Try sudo sysctl -w kernel.perf_event_paranoid=1\n";
                if (fd_refill != -1) close(fd_refill);
                if (fd_access != -1) close(fd_access);
                conv_fn();
                return result;
    }
    // reset, enable both counters 
    ioctl(fd_refill, PERF_EVENT_IOC_RESET, 0);
    ioctl(fd_access, PERF_EVENT_IOC_RESET, 0);
    ioctl(fd_refill, PERF_EVENT_IOC_ENABLE, 0);
    ioctl(fd_access, PERF_EVENT_IOC_ENABLE, 0);

    conv_fn(); //ONLY MEASURE THE KERNEL

    // stop counter
    ioctl(fd_refill, PERF_EVENT_IOC_DISABLE, 0);
    ioctl(fd_access, PERF_EVENT_IOC_DISABLE, 0);

    read(fd_refill, &result.llc_misses, sizeof(result.llc_misses));
    read(fd_access, &result.llc_accesses, sizeof(result.llc_accesses));

    close(fd_refill);
    close(fd_access);
    std::cout << "[perf] l3d_cache_refill (DRAM fetches): " << result.llc_accesses << "\n"
              << "[perf] l3d_cache_rd    (L3 accesses):   " << result.llc_misses << "\n";
    if (result.llc_accesses > 0){
        double miss_rate = 100.0 * result.llc_misses / result.llc_accesses;
        std::cout << "[perf] L3 miss rate: " << miss_rate << " %\n";
    }
    return result;
};

