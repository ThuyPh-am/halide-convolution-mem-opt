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
#include <vector>

// Evict L2/L3 right before counting: touch a buffer larger than the 2 MB L3
static void flush_cache()
{
    static constexpr size_t FLUSH_SIZE = 4 * 1024 * 1024;
    std::vector<char> flush_buf(FLUSH_SIZE, 0);
    for (size_t i = 0; i < FLUSH_SIZE; i += 64)
        flush_buf[i] = i & 0xFF;
    volatile char x = flush_buf[FLUSH_SIZE - 1];
    (void)x;
}

// ARM Cortex A76 PMU event numbers (ARMv8 architecture)
// Source: ARM Architecture Reference Manual / Cortex A76 TRM
static constexpr uint32_t ARM_PMU_L3D_CACHE_REFILL = 0x2a; // l3d_cache_refill
static constexpr uint32_t ARM_PMU_L3D_CACHE = 0xa0;        // (l3d_cache_rd)
static constexpr uint32_t ARM_PMU_L2D_CACHE_REFILL = 0x17;
static constexpr uint32_t ARM_PMU_L2D_CACHE_REFILL_RD = 0x52;
static constexpr uint32_t ARM_PMU_L2D_CACHE_REFILL_WR = 0x53;
static constexpr uint32_t ARM_PMU_L2D_CACHE_REFILL_WB = 0x18;

static int open_raw_pmu_event(uint32_t event_id)
{
    struct perf_event_attr attr;
    memset(&attr, 0, sizeof(attr));
    attr.type = PERF_TYPE_RAW;
    attr.size = sizeof(attr);
    attr.config = event_id;
    attr.disabled = 1;
    attr.exclude_kernel = 0; // include kernel - l3 refills can be kernel-inititated
    attr.exclude_hv = 1;

    int fd = syscall(SYS_perf_event_open, &attr,
                     0,  // current process
                     -1, // any CPU
                     -1, // no group
                     0);
    if (fd == -1)
    {
        std::cerr << "[perf] perf_event_open failed for event 0x"
                  << std::hex << event_id << std::dec
                  << ": " << strerror(errno) << "\n";
    }

    return fd; // -1 on failure
}

PerfResult measure_convolution(std::function<void()> conv_fn)
{
    PerfResult result;
    result.llc_misses = -1; // measurement unavailable

    flush_cache(); // after data generation + JIT compile, before counters

    int fd_refill = open_raw_pmu_event(ARM_PMU_L3D_CACHE_REFILL);
    int fd_access = open_raw_pmu_event(ARM_PMU_L3D_CACHE);
    int l2_refill = open_raw_pmu_event(ARM_PMU_L2D_CACHE_REFILL);
    int l2_rd = open_raw_pmu_event(ARM_PMU_L2D_CACHE_REFILL_RD);
    int l2_wr = open_raw_pmu_event(ARM_PMU_L2D_CACHE_REFILL_WR);
    int l2_wb = open_raw_pmu_event(ARM_PMU_L2D_CACHE_REFILL_WB);

    // rum the kernel even if counters are unavailable
    if (fd_refill == -1 || fd_access == -1 || l2_refill == -1 || l2_rd == -1 || l2_wr == -1 || l2_wb == -1)
    {
        std::cerr << "[perf] Falling back to mock mode (no LLC counts). \n"
                  << "          Try sudo sysctl -w kernel.perf_event_paranoid=1\n";
        if (fd_refill != -1)
            close(fd_refill);
        if (fd_access != -1)
            close(fd_access);
        if (l2_refill != -1)
            close(l2_refill);
        if (l2_rd != -1)
            close(l2_rd);
        if (l2_wr != -1)
            close(l2_wr);
        if (l2_wb != -1)
            close(l2_wb);
        conv_fn();
        return result;
    }
    // reset, enable both counters
    ioctl(fd_refill, PERF_EVENT_IOC_RESET, 0);
    ioctl(fd_refill, PERF_EVENT_IOC_ENABLE, 0);
    ioctl(fd_access, PERF_EVENT_IOC_RESET, 0);
    ioctl(fd_access, PERF_EVENT_IOC_ENABLE, 0);
    ioctl(l2_refill, PERF_EVENT_IOC_RESET, 0);
    ioctl(l2_refill, PERF_EVENT_IOC_ENABLE, 0);
    ioctl(l2_rd, PERF_EVENT_IOC_RESET, 0);
    ioctl(l2_rd, PERF_EVENT_IOC_ENABLE, 0);
    ioctl(l2_wr, PERF_EVENT_IOC_RESET, 0);
    ioctl(l2_wr, PERF_EVENT_IOC_ENABLE, 0);
    ioctl(l2_wb, PERF_EVENT_IOC_RESET, 0);
    ioctl(l2_wb, PERF_EVENT_IOC_ENABLE, 0);
    std::cerr << "[perf] counters enabled, starting kernel\n";

    conv_fn(); // ONLY MEASURE THE KERNEL
    std::cerr << "[perf] kernel done, disabling counters\n";

    // stop counter
    ioctl(fd_refill, PERF_EVENT_IOC_DISABLE, 0);
    ioctl(fd_access, PERF_EVENT_IOC_DISABLE, 0);
    ioctl(l2_refill, PERF_EVENT_IOC_DISABLE, 0);
    ioctl(l2_rd, PERF_EVENT_IOC_DISABLE, 0);
    ioctl(l2_wr, PERF_EVENT_IOC_DISABLE, 0);
    ioctl(l2_wb, PERF_EVENT_IOC_DISABLE, 0);

    read(fd_refill, &result.llc_misses, sizeof(result.llc_misses));
    read(fd_access, &result.llc_accesses, sizeof(result.llc_accesses));
    read(l2_refill, &result.l2_refill, sizeof(result.l2_refill));
    read(l2_rd, &result.l2_read, sizeof(result.l2_read));
    read(l2_wr, &result.l2_write, sizeof(result.l2_write));
    read(l2_wb, &result.l2_write_back, sizeof(result.l2_write_back));

    close(fd_refill);
    close(fd_access);
    close(l2_refill);
    close(l2_rd);
    close(l2_wr);
    close(l2_wb);

    std::cout << "[perf] l3d_cache_refill (DRAM fetches): " << result.llc_misses << "\n"
              << "[perf] l3d_cache_rd    (L3 accesses):   " << result.llc_accesses << "\n"
              << "[perf] l2d_cache_refill    (L2 accesses):   " << result.l2_refill << "\n"
              << "[perf] l2d_cache_refill_rd    (L2 accesses):   " << result.l2_read << "\n"
              << "[perf] l2d_cache_refill_wr    (L2 accesses):   " << result.l2_write << "\n"
              << "[perf] l2d_cache_wb (L2 write back):  " << result.l2_write_back << "\n";
    if (result.llc_accesses > 0)
    {
        double miss_rate = 100.0 * result.llc_misses / result.llc_accesses;
        double l2_miss_rate = 100.0 * result.l2_refill / result.l2_read;
        std::cout << "[perf] L3 miss rate: " << miss_rate << " %\n"
                  << "[perf] L2 miss rate: " << l2_miss_rate << " %\n";
    }
    return result;
};
