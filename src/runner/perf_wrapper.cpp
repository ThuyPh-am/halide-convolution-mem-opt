// Wraps a kernel execution with Linux perf_event_open syscall to measure
// LLC-load-misses (proxy for DRAM accesses) and wall-clock time.
// Design to isolated convolution counter
#include "perf_wrapper.h"
#include <linux/perf_event.h>
#include <sys/ioctl.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <iostream>
#include <functional>
#include <cstring>


static long perf_event_open(struct perf_event_attr *hw_event, pid_t pid, int cpu,
                            int group_fd, unsigned long flags){
    int fd = syscall(SYS_perf_event_open, hw_event, pid, cpu, group_fd, flags);
    if (fd = -1){
        std::cerr << "Error creating event"<< std::endl;
        // exit(EXIT_FAILURE); 
        // modify to use mock case 
        return -2;
    }

    return fd;
};

PerfResult measure_convolution(std::function<void()>conv_fn){
    int fd;
    PerfResult measure;

    // configure the perf_event_attr to define which event to be counted
    struct perf_event_attr attr;
    memset(&attr, 0, sizeof(attr));
    attr.type = PERF_TYPE_HW_CACHE;
    attr.size = sizeof(struct perf_event_attr);
    attr.config = (PERF_COUNT_HW_CACHE_LL) | 
                    (PERF_COUNT_HW_CACHE_OP_READ << 8) |
                    (PERF_COUNT_HW_CACHE_RESULT_MISS << 16);
    attr.disabled = 1;

    // access counter
    fd = perf_event_open(&attr, 0, -1, -1, 0);

    // mock case
    if (fd == -2){
        conv_fn();
        measure.llc_misses == -1 ;
        return measure;
    }
    // start counter
    ioctl(fd, PERF_EVENT_IOC_RESET, 0);
    ioctl(fd, PERF_EVENT_IOC_ENABLE, 0);

    // 
    conv_fn();
    // stop counter
    ioctl(fd, PERF_EVENT_IOC_DISABLE, 0);
    read(fd, &measure.llc_misses, sizeof(measure.llc_misses));
    std::cout << "LLC load misses" << measure.llc_misses << std::endl;
    close(fd);
    return measure;
};

