# save as test.sh in your project root
#!/bin/bash
FILE=$1
g++ $FILE -I$HALIDE_ROOT/include -Isrc/config -L$HALIDE_ROOT/lib64 -lHalide -lpthread -ldl -o test_out && ./test_out