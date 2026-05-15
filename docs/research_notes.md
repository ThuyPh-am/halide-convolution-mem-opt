The paper title: 
* Eyeriss: An Energy-Efficient Reconfigurable Accelerator for Deep Convolutional Neural Networks


Models benchmark: (p.2)
* AlexNet
* VGG - 16


The input frames are resized according to (p.9)
* AlexNet: 227 × 227 and batch size 4
* VGG-16: 224 × 224 and batch size 3


Dataset: 
* 1000-class image classification on the ImageNet data set (p. 2)




* Halide:
** If not include Halide name space, I must write Halide::... and load helper function like load_image
** Can write function, declare varaible but the variables or functionc or expression don't know their names so it is easier for debug to pass the name as str
** Halide is a feed forward graphs of funcs, it does not allow loop within pipelines, only func with a given definition can be called.
** Halide::Func gradient ("gradient");
** Halide::Buffer<uint_8> input = load_image(Path);
** Halide::Buffer<uint_8> output = gradient.realize({8,8}); => realize the function gradient with domain {height, width}
** gradient.trace_stores(); => to print out the computation of algorithm
** gradient.print_loop_nest();

** Halide schedule:
*** the default order is row major traverse (y moves slowlier, x moves faster)
*** gradient.reorder(y,x);
*** gradient.split(x,x_outer, x_inner, 4); => x outer run from 0 to the closest larger multiple of factor (4 in this case)
for example {7,2} splitting factor 3, than x will be proceed in group [0,1,2] [3,4,5] [4,5,6]
for pure halide, there is no side effects so re-compute causes no harm

*** gradient.fuse(x,y,fused)
*** gradient.split(x,x_outer, x_inner, 4)
            .split(y, y_outer, y_inner, 4)
            .reorder(x_inner, y_inner, x_outer, y_outer);
    
    OR 
    gradient.tile(x,y,x_outer, y_outer, x_inner, y_inner, 4, 4);

*** vectors: processing in vector  instead of splitting then processing with loop 
gradient.split(x, x_outer, x_inner, 4);
gradient.vectorize(x_inner);
or 
gradient.vectorize(x,4); (uisng this notation is equivalent with split x_outer and x_inner but call x_outer as x) 
x will be processed as [0,1,2,3] + [0,0,0,0] instead of for loop

*** unrolling a loop: remove loop, write the function explicitly repeatedly, so chances for overlapped data being notified 
gradient.split(x,x_outer, x_inner, 2);
gradient.unroll(x_inner);
or
gradient.unroll(x,2);


*** fusing, tilling, parallelizing
fusing works better than nested parallel loop, and the execution order is NOT guarantee in parallel 
gradient.tile(x, y, x_outer, y_outer, x_inner, y_inner, 4, 4)
        .fuse(x_outer, x_outer, tile_index)
        .parallel(tile_index);


*** fusing,tilling, parallelizing, unrolling, vectorizing
it is possible to tile, fuse, then parallel execute AND tile into sub-tiles, vectorize x, and unroll y

var x_outer, y_outter, x_inner, y_inner, tile_index;
gradient.tile(x, y, x_outer, y_outer, x_inner, y_inner, 64, 64)
        .fuse(x_outer, y_outer, tile_index)
        .parallel(tile_index);

var x_inner_outer, y_inner_outer, x_vectors, y_pairs;
gradient.tile(x_inner, y_inner, x_inner_outer, y_inner_outer, x_vectors, y_pairs, 4, 2)
        .vectorize(x_vectors)
        .unrol(y_pairs);

// approximation of RS on CPU:
// 1. A tile of output rows stays in L1/L2 while i iterate over
    // every filter row and every input channel that feeds it -> input-row reuse
// 2. The reduction is ordered r[2] (C_in) -> r[1] (kH) -> r[0] [kW]
    // so the innermost access pattern matches RS: for a fixed filter row,
    // sweep the input row spatially (kW), then advance to the next filter row
// 3. Output tiles are vectorised over W_out to exploit SIMD and keep prtial
    // sum in registers during the ineermost loop

