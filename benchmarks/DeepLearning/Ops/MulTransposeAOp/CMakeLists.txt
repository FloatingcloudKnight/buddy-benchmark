# Add the executable for the Conv2D NHWC-FHWC benchmark
add_executable(dl-op-mul-transpose-a-benchmark
  Main.cpp
)
target_link_libraries(dl-op-mul-transpose-a-benchmark GoogleBenchmark)

# CMAKE_C_FLAGS is set when configuring CMake
separate_arguments(CLANG_FLAGS_LIST UNIX_COMMAND "${CMAKE_C_FLAGS}")

# -----------------------------------------------------------------------------
# Build and link mul_transpose_a_scalar_O0.o
# -----------------------------------------------------------------------------
add_custom_command(OUTPUT mul_transpose_a_scalar_O0.o
  COMMAND cat ${CMAKE_CURRENT_SOURCE_DIR}/MulTransposeA.mlir |
      sed -e {s/@mul_transpose_a/@mul_transpose_a_scalar_O0/} |
      ${LLVM_MLIR_BINARY_DIR}/mlir-opt
        -pass-pipeline 
        "builtin.module(func.func(tosa-to-linalg-named),func.func(tosa-to-linalg),func.func(tosa-to-tensor),func.func(tosa-to-arith))" |
      ${BUDDY_MLIR_BINARY_DIR}/buddy-opt
        -arith-expand
        -eliminate-empty-tensors
        -empty-tensor-to-alloc-tensor
        -one-shot-bufferize
        -convert-linalg-to-affine-loops
        -affine-loop-fusion
        -affine-parallelize
        -lower-affine
        -func-bufferize
        -arith-bufferize
        -tensor-bufferize
        -buffer-deallocation
        -finalizing-bufferize
        -convert-vector-to-scf
        -expand-strided-metadata
        -convert-vector-to-llvm
        -memref-expand
        -arith-expand
        -convert-arith-to-llvm
        -finalize-memref-to-llvm
        -convert-scf-to-cf
        -llvm-request-c-wrappers
        -convert-arith-to-llvm
        -convert-math-to-llvm
        -convert-math-to-libm
        -convert-func-to-llvm
        -reconcile-unrealized-casts |
      ${LLVM_MLIR_BINARY_DIR}/mlir-translate --mlir-to-llvmir -o mul_transpose_a_scalar_O0.ll
  COMMAND ${LLVM_MLIR_BINARY_DIR}/clang -O0 ${CLANG_FLAGS_LIST} mul_transpose_a_scalar_O0.ll
          -c -save-temps -o ${CMAKE_CURRENT_BINARY_DIR}/mul_transpose_a_scalar_O0.o
  VERBATIM)

add_library(mul_transpose_a_scalar_O0 STATIC mul_transpose_a_scalar_O0.o)
set_target_properties(mul_transpose_a_scalar_O0 PROPERTIES LINKER_LANGUAGE CXX)
target_link_libraries(dl-op-mul-transpose-a-benchmark
  mul_transpose_a_scalar_O0
)

# -----------------------------------------------------------------------------
# Build and link mul_transpose_a_scalar_O3.o
# -----------------------------------------------------------------------------
add_custom_command(OUTPUT mul_transpose_a_scalar_O3.o
  COMMAND cat ${CMAKE_CURRENT_SOURCE_DIR}/MulTransposeA.mlir |
      sed -e {s/@mul_transpose_a/@mul_transpose_a_scalar_O3/} |
      ${LLVM_MLIR_BINARY_DIR}/mlir-opt
        -pass-pipeline 
        "builtin.module(func.func(tosa-to-linalg-named),func.func(tosa-to-linalg),func.func(tosa-to-tensor),func.func(tosa-to-arith))" |
      ${BUDDY_MLIR_BINARY_DIR}/buddy-opt
        -arith-expand
        -eliminate-empty-tensors
        -empty-tensor-to-alloc-tensor
        -one-shot-bufferize
        -convert-linalg-to-affine-loops
        -affine-loop-fusion
        -affine-parallelize
        -lower-affine
        -func-bufferize
        -arith-bufferize
        -tensor-bufferize
        -buffer-deallocation
        -finalizing-bufferize
        -convert-vector-to-scf
        -expand-strided-metadata
        -convert-vector-to-llvm
        -memref-expand
        -arith-expand
        -convert-arith-to-llvm
        -finalize-memref-to-llvm
        -convert-scf-to-cf
        -llvm-request-c-wrappers
        -convert-arith-to-llvm
        -convert-math-to-llvm
        -convert-math-to-libm
        -convert-func-to-llvm
        -reconcile-unrealized-casts |
      ${LLVM_MLIR_BINARY_DIR}/mlir-translate --mlir-to-llvmir -o mul_transpose_a_scalar_O3.ll
  COMMAND ${LLVM_MLIR_BINARY_DIR}/clang -O3 ${CLANG_FLAGS_LIST} mul_transpose_a_scalar_O3.ll
          -c -save-temps -o ${CMAKE_CURRENT_BINARY_DIR}/mul_transpose_a_scalar_O3.o
  VERBATIM)

add_library(mul_transpose_a_scalar_O3 STATIC mul_transpose_a_scalar_O3.o)
set_target_properties(mul_transpose_a_scalar_O3 PROPERTIES LINKER_LANGUAGE CXX)
target_link_directories(dl-op-mul-transpose-a-benchmark PRIVATE ${LLVM_MLIR_LIBRARY_DIR})
target_link_libraries(dl-op-mul-transpose-a-benchmark
  mul_transpose_a_scalar_O3
)

# -----------------------------------------------------------------------------
# Build and link mul_transpose_a_vectorization.o
# -----------------------------------------------------------------------------
add_custom_command(OUTPUT mul_transpose_a_vectorization.o
  COMMAND cat ${CMAKE_CURRENT_SOURCE_DIR}/MulTransposeVec.mlir |
      sed -e {s/@mul_transpose_a/@mul_transpose_a_vectorization/} |
      ${BUDDY_MLIR_BINARY_DIR}/buddy-opt
        -arith-expand
        -eliminate-empty-tensors
        -empty-tensor-to-alloc-tensor
        -one-shot-bufferize
        -func-bufferize
        -arith-bufferize
        -convert-linalg-to-loops
        -affine-loop-fusion
        -lower-affine
        -convert-vector-to-scf
        -expand-strided-metadata
        -convert-vector-to-llvm
        -memref-expand
        -arith-expand
        -convert-arith-to-llvm
        -finalize-memref-to-llvm
        -convert-scf-to-cf
        -llvm-request-c-wrappers
        -convert-arith-to-llvm
        -convert-math-to-llvm
        -convert-math-to-libm
        -convert-func-to-llvm
        -reconcile-unrealized-casts |
      ${LLVM_MLIR_BINARY_DIR}/mlir-translate --mlir-to-llvmir -o mul_transpose_a_vectorization.ll
  COMMAND ${LLVM_MLIR_BINARY_DIR}/clang -O3 ${CLANG_FLAGS_LIST} mul_transpose_a_vectorization.ll
          -c -save-temps -o ${CMAKE_CURRENT_BINARY_DIR}/mul_transpose_a_vectorization.o
  VERBATIM)

add_library(mul_transpose_a_vectorization STATIC mul_transpose_a_vectorization.o)
set_target_properties(mul_transpose_a_vectorization PROPERTIES LINKER_LANGUAGE CXX)
target_link_libraries(dl-op-mul-transpose-a-benchmark
  mul_transpose_a_vectorization
)
