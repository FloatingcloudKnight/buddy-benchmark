# add_custom_command(
#   OUTPUT
#     ${CMAKE_CURRENT_SOURCE_DIR}/forward.mlir
#     ${CMAKE_CURRENT_SOURCE_DIR}/subgraph0.mlir
#   COMMAND
#     python3 ${CMAKE_CURRENT_SOURCE_DIR}/buddy_rmsnorm_import.py
#   COMMENT "Generating forward.mlir, subgraph0.mlir"
# )

add_executable(dl-model-llama-benchmark
  Main.cpp
)

set(LLAMA_EXAMPLE_PATH ${CMAKE_CURRENT_SOURCE_DIR})

target_compile_definitions(dl-model-llama-benchmark PRIVATE 
  LLAMA_EXAMPLE_PATH="${LLAMA_EXAMPLE_PATH}" 
)

# target_link_libraries(dl-model-llama-benchmark
#   GoogleBenchmark
# )

separate_arguments(CLANG_FLAGS_LIST UNIX_COMMAND "${CMAKE_C_FLAGS}")

add_custom_command(
  OUTPUT forward_scalar.o
  COMMAND cat ${CMAKE_CURRENT_SOURCE_DIR}/forward.mlir |
    sed -e {s/@forward/@forward_scalar/} |
    sed -e {s/@subgraph0/@subgraph0_scalar/} |
  ${LLVM_MLIR_BINARY_DIR}/mlir-opt
    -expand-strided-metadata
    -finalize-memref-to-llvm
    -llvm-request-c-wrappers
    -convert-func-to-llvm
    -reconcile-unrealized-casts |
  ${LLVM_MLIR_BINARY_DIR}/mlir-translate -mlir-to-llvmir -o forward_scalar.ll
  COMMAND ${LLVM_MLIR_BINARY_DIR}/clang -O3 ${CLANG_FLAGS_LIST} forward_scalar.ll
    -c -save-temps -o ${CMAKE_CURRENT_BINARY_DIR}/forward_scalar.o
  DEPENDS
    ${CMAKE_CURRENT_SOURCE_DIR}/forward.mlir
  COMMENT "Building forward_scalar.o"
  VERBATIM
)

add_custom_command(
  OUTPUT subgraph0_scalar.o
  COMMAND cat ${CMAKE_CURRENT_SOURCE_DIR}/subgraph0.mlir |
    sed -e {s/@subgraph0/@subgraph0_scalar/} |
    ${LLVM_MLIR_BINARY_DIR}/mlir-opt
      -pass-pipeline
      "builtin.module(func.func(tosa-to-linalg-named),func.func(tosa-to-linalg),func.func(tosa-to-tensor),func.func(tosa-to-arith))" |
    ${BUDDY_MLIR_BINARY_DIR}/buddy-opt
      -convert-elementwise-to-linalg
      -arith-expand
      -eliminate-empty-tensors
      -empty-tensor-to-alloc-tensor
      -one-shot-bufferize
      -func-bufferize-dynamic-offset
      -tensor-bufferize
      -arith-bufferize
      -buffer-deallocation
      -finalizing-bufferize
      -matmul-parallel-vectorization-optimize
      -batchmatmul-optimize
      -convert-linalg-to-affine-loops
      -affine-loop-fusion
      -affine-parallelize
      -lower-affine
      -convert-scf-to-openmp
      -convert-vector-to-scf
      -expand-strided-metadata
      -cse
      -convert-vector-to-llvm
      -memref-expand
      -arith-expand
      -convert-arith-to-llvm
      -finalize-memref-to-llvm
      -convert-scf-to-cf
      -llvm-request-c-wrappers
      -convert-openmp-to-llvm
      -convert-arith-to-llvm
      -convert-math-to-llvm
      -convert-math-to-libm 
      -convert-func-to-llvm
      -reconcile-unrealized-casts |
  ${LLVM_MLIR_BINARY_DIR}/mlir-translate -mlir-to-llvmir -o subgraph0_scalar.ll
  COMMAND ${LLVM_MLIR_BINARY_DIR}/clang -O3 ${CLANG_FLAGS_LIST} subgraph0_scalar.ll
    -c -save-temps -o ${CMAKE_CURRENT_BINARY_DIR}/subgraph0_scalar.o
  DEPENDS
    ${CMAKE_CURRENT_SOURCE_DIR}/subgraph0.mlir
  COMMENT "Building subgraph0_scalar.o"
  VERBATIM)

add_library(llama_scalar subgraph0_scalar.o forward_scalar.o)
set_target_properties(llama_scalar PROPERTIES LINKER_LANGUAGE CXX)

target_link_libraries(dl-model-llama-benchmark
  ${OMP_SHARED_DIR}
  llama_scalar
  ${BUDDY_LIB_DIR}/libStaticMLIRCRunnerUtils.a
)
