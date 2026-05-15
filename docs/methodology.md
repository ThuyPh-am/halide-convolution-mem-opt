Halide lets you choose:

* Inline everything (minimal memory, more math)

* Compute everything at root (minimal math, more memory)

* Or something in between

=> the question is: is math or memory expensive? 
* when doing redundant math: full inlining. -> Uses little memory

                                            -> Uses little memory traffic

                                            -> But does a lot of repeated math
* when compute everything first: compute_root(). -> Uses more memory

                                                -> But does less math

* there is the possibility to use compute_at(consumer,y) -> to execute the computation according to
                                                             y coordinates of the consumer program
                                store_root() -> to buffer and store at the outermost level
                                So Are We Using Full Parallelism?
                                Yes — in compute. 
                                Each thread still works independently.
                                But not fully optimized memory-wise.
                                Because:
                                * Halide disables circular buffering
                                * Halide avoids reuse tricks
                                * Memory footprint grows
                                * So the bottleneck may shift to memory bandwidth.

LLVM 19.1.7, Halide ver 18.0.0, raspberry pi 5 8GB (NEON -ARM's SIMD extension, quad core)
L1d cache:                               256 KiB (4 instances)
L1i cache:                               256 KiB (4 instances)
L2 cache:                                2 MiB (4 instances)
L3 cache:                                2 MiB (1 instance)

        Documents  Halide          llvm.sh    mem-opt  Pictures  Templates
Desktop  Downloads  halide-install  llvm.sh.1  Music    Public    Videos

 Performance counter stats for 'ls':

              1.00 msec task-clock:u                     #    0.593 CPUs utilized             
                 0      context-switches:u               #    0.000 /sec                      
                 0      cpu-migrations:u                 #    0.000 /sec                      
                63      page-faults:u                    #   63.314 K/sec                     
           548,062      cycles:u                         #    0.551 GHz                       
           529,272      instructions:u                   #    0.97  insn per cycle            
           110,855      branches:u                       #  111.408 M/sec                     
             7,855      branch-misses:u                  #    7.09% of all branches           

       0.001676941 seconds time elapsed

       0.001737000 seconds user
       0.000000000 seconds sys



** Dataflow techniques inspired loop reorder **
* Baseline loop order * 
for n in batch [0,..., N]:
        for C_out in ofmap channel [0,...,C_out]:
                for y in ofmap height [0,...,H]:
                        for x in ofmap width [0,...,W]:
                                for C_in in ifmap channel [0,..., C_in]:
                                        for ky in kernel height [0,...,k_h]:
                                                for kx in kernel width [0,...,k_w]:
                                                        output[n, c_out, y, x] += ifmap[n, c_in, y+ky, x+kx] * weight[c_out, c_in, ky, kx]
weights read = N * C_out * H * W * C_in * k_H * k_W
ifmap read = N * C_out * H * W * C_in * k_H * k_W
ofmap write = N * C_out * H * W
ofmap read = 0

weight reuse = 1
ifmap reuse = 1
ofmap reuse = C_in * k_h * k_w

* WS inspired loop reorder * 
for C_out in ofmap channel [0,...,C_out]:
        for C_in in ifmap channel [0,...,C_in]:
                for ky in kernel height [0,...,k_h]:
                        for kx in kernel width [0,...,k_w]:
                                for n in batch [0,..., N]:
                                        for y in ofmap height [0,...,H]:
                                                for x in ofmap width [0,...,W]:
                                                        output[n, c_out, y, x] += ifmap[n, c_in, y+ky, x+kx] * weight[c_out, c_in, ky, kx]

weights read = C_out * C_in * k_h * k_w
ifmap read = N * C_in * H * W * k_h * k_w
ofmap write = N * C_out * H * W
ofmap read = 0

weight reuse = N * H * W
ifmap reuse = C_out
ofmap reuse = C_in * k_h * k_w

* OS inspired loop reorder * 
The baseline loop order already have the output indices at the outermost loops, which already keep output stationary
The output read of the baseline is also 0 and the output write is only N * C_out * H * W
the other calculation are similar with baseline
ofmap reuse = C_in * k_h * k_w

* RS inspired loop reorder * 
Theoretically, RS is to have 1 row of weight AND corresponding 1 row of input AND 1 row of psum stationary. Thus:
        1 row of weight : C-out, C-in, ky in the outermost
        1 row of input: N, y in the outermost
        kx and x slide

for n in batch [0,..., N]:
        for C_out in ofmap channel [0,...,C_out]:
                        for C_in in ifmap channel [0,...,C_in]:
                                for ky in kernel height [0,...,k_h]:
                                        for y in ofmap height [0,...,H]:
                                                for kx in kernel width [0,...,k_w]:
                                                        for x in ofmap width [0,...,W]:
                                                                output[n, c_out, y, x] += ifmap[n, c_in, y+ky, x+kx] * weight[c_out, c_in, ky, kx]

Weights read = C_out * C_in * k_h
ifmap read = N * C_in * (H + k_h -1)
psum read = 0
psum write = N * C_out * H

weight reuse = N * W * k_w
ifmap reuse = C_out * W * k_W
psum reuse = W * k_w