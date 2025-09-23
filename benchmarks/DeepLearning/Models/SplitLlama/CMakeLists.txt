
# need modified:
set(Boost_INCLUDE_DIR "/YOUR_BOOST_PATH")
set(Websocket_INCLUDE_DIR "/YOUR_WEBSOCKETPP_PATH")

set(LLAMA_SPLIT_EXAMPLE_PATH "/YOUR_RUNTIME_PATH")
set(LLAMA_EXAMPLE_BUILD_PATH "/YOUR_RUNTIME_PATH")

set(DEVICE1 "fd7a:115c:a1e0::7501:f32e")
set(DEVICE2 "fd7a:115c:a1e0::9b01:8a0b")
set(DEVICE3 "fd7a:115c:a1e0::af01:da5e")
set(DEVICE4 "fd7a:115c:a1e0::a101:de77")
set(DEVICE5 "fd7a:115c:a1e0::501:b86f")
set(DEVICE6 "fd7a:115c:a1e0::ff01:7962")
set(DEVICE7 "fd7a:115c:a1e0::f601:2432")
set(DEVICE8 "fd7a:115c:a1e0::f301:db3f")
set(DEVICE9 "fd7a:115c:a1e0::1101:3f1b")
set(PORTS 9001 9002 9003 9004 9005 9006 9007 9008 9009 9010 9011 9012 9013 9014)

set(PART_NUMS 2)
set(RUNNER_IDS 0 1)




separate_arguments(CLANG_FLAGS_LIST UNIX_COMMAND "${CMAKE_C_FLAGS}")
set(Boost_USE_STATIC_LIBS ON)
set(Boost_USE_STATIC_RUNTIME ON)

add_custom_command(
  OUTPUT forward0.o
  COMMAND ${LLVM_MLIR_BINARY_DIR}/mlir-opt ${CMAKE_CURRENT_SOURCE_DIR}/graph/forward0.mlir
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named),func.func(tosa-to-linalg),func.func(tosa-to-tensor),func.func(tosa-to-arith))" |
          ${BUDDY_MLIR_BINARY_DIR}/buddy-opt
            -arith-expand
            -eliminate-empty-tensors
            -empty-tensor-to-alloc-tensor
            -one-shot-bufferize
            -matmul-parallel-vectorization-optimize
            -batchmatmul-optimize
            -convert-linalg-to-affine-loops
            -affine-loop-fusion
            -affine-parallelize
            -lower-affine
            -convert-scf-to-openmp
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
            -convert-openmp-to-llvm
            -convert-arith-to-llvm
            -convert-math-to-llvm
            -convert-math-to-libm 
            -convert-func-to-llvm
            -reconcile-unrealized-casts |
          ${LLVM_MLIR_BINARY_DIR}/mlir-translate -mlir-to-llvmir -o forward_scalar.ll
  COMMAND ${LLVM_MLIR_BINARY_DIR}/clang -O3 ${CLANG_FLAGS_LIST} forward_scalar.ll
    -c -save-temps -o ${CMAKE_CURRENT_BINARY_DIR}/forward0.o
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/graph/forward0.mlir
  COMMENT "Building forward0.o "
  VERBATIM)

add_custom_command(
    OUTPUT subgraph0.o
    COMMAND ${LLVM_MLIR_BINARY_DIR}/mlir-opt ${CMAKE_CURRENT_SOURCE_DIR}/graph/subgraph0.mlir 
              -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named),func.func(tosa-to-linalg),func.func(tosa-to-tensor),func.func(tosa-to-arith))" |
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
    -c -save-temps -o ${CMAKE_CURRENT_BINARY_DIR}/subgraph0.o
    DEPENDS 
      ${CMAKE_CURRENT_SOURCE_DIR}/graph/subgraph0.mlir
    COMMENT "Building subgraph0.o "
    VERBATIM)

add_library(SPLITLLAMA0 STATIC forward0.o subgraph0.o)


add_custom_command(
  OUTPUT forward1.o
  COMMAND ${LLVM_MLIR_BINARY_DIR}/mlir-opt ${CMAKE_CURRENT_SOURCE_DIR}/graph/forward1.mlir
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named),func.func(tosa-to-linalg),func.func(tosa-to-tensor),func.func(tosa-to-arith))" |
          ${BUDDY_MLIR_BINARY_DIR}/buddy-opt
            -arith-expand
            -eliminate-empty-tensors
            -empty-tensor-to-alloc-tensor
            -one-shot-bufferize
            -matmul-parallel-vectorization-optimize
            -batchmatmul-optimize
            -convert-linalg-to-affine-loops
            -affine-loop-fusion
            -affine-parallelize
            -lower-affine
            -convert-scf-to-openmp
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
            -convert-openmp-to-llvm
            -convert-arith-to-llvm
            -convert-math-to-llvm
            -convert-math-to-libm 
            -convert-func-to-llvm
            -reconcile-unrealized-casts |
          ${LLVM_MLIR_BINARY_DIR}/mlir-translate -mlir-to-llvmir -o forward1.ll
  COMMAND ${LLVM_MLIR_BINARY_DIR}/clang -O3 ${CLANG_FLAGS_LIST} forward1.ll
    -c -save-temps -o ${CMAKE_CURRENT_BINARY_DIR}/forward1.o
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/graph/forward1.mlir
  COMMENT "Building forward1.o "
  VERBATIM)

add_custom_command(
    OUTPUT subgraph1.o
    COMMAND ${LLVM_MLIR_BINARY_DIR}/mlir-opt ${CMAKE_CURRENT_SOURCE_DIR}/graph/subgraph1.mlir 
              -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named),func.func(tosa-to-linalg),func.func(tosa-to-tensor),func.func(tosa-to-arith))" |
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
            ${LLVM_MLIR_BINARY_DIR}/mlir-translate -mlir-to-llvmir -o subgraph1.ll
  COMMAND ${LLVM_MLIR_BINARY_DIR}/clang -O3 ${CLANG_FLAGS_LIST} subgraph1.ll
    -c -save-temps -o ${CMAKE_CURRENT_BINARY_DIR}/subgraph1.o
    DEPENDS 
      ${CMAKE_CURRENT_SOURCE_DIR}/graph/subgraph1.mlir
    COMMENT "Building subgraph1.o "
    VERBATIM)

add_library(SPLITLLAMA1 STATIC forward1.o subgraph1.o)

add_custom_command(
  OUTPUT forward2.o
  COMMAND ${LLVM_MLIR_BINARY_DIR}/mlir-opt ${CMAKE_CURRENT_SOURCE_DIR}/graph/forward2.mlir
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named),func.func(tosa-to-linalg),func.func(tosa-to-tensor),func.func(tosa-to-arith))" |
          ${BUDDY_MLIR_BINARY_DIR}/buddy-opt
            -arith-expand
            -eliminate-empty-tensors
            -empty-tensor-to-alloc-tensor
            -one-shot-bufferize
            -matmul-parallel-vectorization-optimize
            -batchmatmul-optimize
            -convert-linalg-to-affine-loops
            -affine-loop-fusion
            -affine-parallelize
            -lower-affine
            -convert-scf-to-openmp
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
            -convert-openmp-to-llvm
            -convert-arith-to-llvm
            -convert-math-to-llvm
            -convert-math-to-libm 
            -convert-func-to-llvm
            -reconcile-unrealized-casts |
          ${LLVM_MLIR_BINARY_DIR}/mlir-translate -mlir-to-llvmir -o forward2.ll
  COMMAND ${LLVM_MLIR_BINARY_DIR}/clang -O3 ${CLANG_FLAGS_LIST} forward2.ll
    -c -save-temps -o ${CMAKE_CURRENT_BINARY_DIR}/forward2.o
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/graph/forward2.mlir
  COMMENT "Building forward2.o "
  VERBATIM)

add_custom_command(
    OUTPUT subgraph2.o
    COMMAND ${LLVM_MLIR_BINARY_DIR}/mlir-opt ${CMAKE_CURRENT_SOURCE_DIR}/graph/subgraph2.mlir 
              -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named),func.func(tosa-to-linalg),func.func(tosa-to-tensor),func.func(tosa-to-arith))" |
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
            ${LLVM_MLIR_BINARY_DIR}/mlir-translate -mlir-to-llvmir -o subgraph2.ll
  COMMAND ${LLVM_MLIR_BINARY_DIR}/clang -O3 ${CLANG_FLAGS_LIST} subgraph2.ll
    -c -save-temps -o ${CMAKE_CURRENT_BINARY_DIR}/subgraph2.o
    DEPENDS 
      ${CMAKE_CURRENT_SOURCE_DIR}/graph/subgraph2.mlir
    COMMENT "Building subgraph2.o "
    VERBATIM)

add_library(SPLITLLAMA2 STATIC forward2.o subgraph2.o)

add_custom_command(
  OUTPUT forward3.o
  COMMAND ${LLVM_MLIR_BINARY_DIR}/mlir-opt ${CMAKE_CURRENT_SOURCE_DIR}/graph/forward3.mlir
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named),func.func(tosa-to-linalg),func.func(tosa-to-tensor),func.func(tosa-to-arith))" |
          ${BUDDY_MLIR_BINARY_DIR}/buddy-opt
            -arith-expand
            -eliminate-empty-tensors
            -empty-tensor-to-alloc-tensor
            -one-shot-bufferize
            -matmul-parallel-vectorization-optimize
            -batchmatmul-optimize
            -convert-linalg-to-affine-loops
            -affine-loop-fusion
            -affine-parallelize
            -lower-affine
            -convert-scf-to-openmp
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
            -convert-openmp-to-llvm
            -convert-arith-to-llvm
            -convert-math-to-llvm
            -convert-math-to-libm 
            -convert-func-to-llvm
            -reconcile-unrealized-casts |
          ${LLVM_MLIR_BINARY_DIR}/mlir-translate -mlir-to-llvmir -o forward3.ll
  COMMAND ${LLVM_MLIR_BINARY_DIR}/clang -O3 ${CLANG_FLAGS_LIST} forward3.ll
    -c -save-temps -o ${CMAKE_CURRENT_BINARY_DIR}/forward3.o
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/graph/forward3.mlir
  COMMENT "Building forward3.o "
  VERBATIM)

add_custom_command(
    OUTPUT subgraph3.o
    COMMAND ${LLVM_MLIR_BINARY_DIR}/mlir-opt ${CMAKE_CURRENT_SOURCE_DIR}/graph/subgraph3.mlir 
              -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named),func.func(tosa-to-linalg),func.func(tosa-to-tensor),func.func(tosa-to-arith))" |
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
            ${LLVM_MLIR_BINARY_DIR}/mlir-translate -mlir-to-llvmir -o subgraph3.ll
  COMMAND ${LLVM_MLIR_BINARY_DIR}/clang -O3 ${CLANG_FLAGS_LIST} subgraph3.ll
    -c -save-temps -o ${CMAKE_CURRENT_BINARY_DIR}/subgraph3.o
    DEPENDS 
      ${CMAKE_CURRENT_SOURCE_DIR}/graph/subgraph3.mlir
    COMMENT "Building subgraph3.o "
    VERBATIM)

add_library(SPLITLLAMA3 STATIC forward3.o subgraph3.o)

add_custom_command(
  OUTPUT forward5.o
  COMMAND ${LLVM_MLIR_BINARY_DIR}/mlir-opt ${CMAKE_CURRENT_SOURCE_DIR}/graph/forward5.mlir
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named),func.func(tosa-to-linalg),func.func(tosa-to-tensor),func.func(tosa-to-arith))" |
          ${BUDDY_MLIR_BINARY_DIR}/buddy-opt
            -arith-expand
            -eliminate-empty-tensors
            -empty-tensor-to-alloc-tensor
            -one-shot-bufferize
            -matmul-parallel-vectorization-optimize
            -batchmatmul-optimize
            -convert-linalg-to-affine-loops
            -affine-loop-fusion
            -affine-parallelize
            -lower-affine
            -convert-scf-to-openmp
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
            -convert-openmp-to-llvm
            -convert-arith-to-llvm
            -convert-math-to-llvm
            -convert-math-to-libm 
            -convert-func-to-llvm
            -reconcile-unrealized-casts |
          ${LLVM_MLIR_BINARY_DIR}/mlir-translate -mlir-to-llvmir -o forward5.ll
  COMMAND ${LLVM_MLIR_BINARY_DIR}/clang -O3 ${CLANG_FLAGS_LIST} forward5.ll
    -c -save-temps -o ${CMAKE_CURRENT_BINARY_DIR}/forward5.o
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/graph/forward5.mlir
  COMMENT "Building forward5.o "
  VERBATIM)

add_custom_command(
    OUTPUT subgraph5.o
    COMMAND ${LLVM_MLIR_BINARY_DIR}/mlir-opt ${CMAKE_CURRENT_SOURCE_DIR}/graph/subgraph5.mlir 
              -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named),func.func(tosa-to-linalg),func.func(tosa-to-tensor),func.func(tosa-to-arith))" |
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
            ${LLVM_MLIR_BINARY_DIR}/mlir-translate -mlir-to-llvmir -o subgraph5.ll
  COMMAND ${LLVM_MLIR_BINARY_DIR}/clang -O3 ${CLANG_FLAGS_LIST} subgraph5.ll
    -c -save-temps -o ${CMAKE_CURRENT_BINARY_DIR}/subgraph5.o
    DEPENDS 
      ${CMAKE_CURRENT_SOURCE_DIR}/graph/subgraph5.mlir
    COMMENT "Building subgraph5.o "
    VERBATIM)

add_library(SPLITLLAMA5 STATIC forward5.o subgraph5.o)

add_custom_command(
  OUTPUT forward193.o
  COMMAND ${LLVM_MLIR_BINARY_DIR}/mlir-opt ${CMAKE_CURRENT_SOURCE_DIR}/graph/forward193.mlir
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named),func.func(tosa-to-linalg),func.func(tosa-to-tensor),func.func(tosa-to-arith))" |
          ${BUDDY_MLIR_BINARY_DIR}/buddy-opt
            -arith-expand
            -eliminate-empty-tensors
            -empty-tensor-to-alloc-tensor
            -one-shot-bufferize
            -matmul-parallel-vectorization-optimize
            -batchmatmul-optimize
            -convert-linalg-to-affine-loops
            -affine-loop-fusion
            -affine-parallelize
            -lower-affine
            -convert-scf-to-openmp
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
            -convert-openmp-to-llvm
            -convert-arith-to-llvm
            -convert-math-to-llvm
            -convert-math-to-libm 
            -convert-func-to-llvm
            -reconcile-unrealized-casts |
          ${LLVM_MLIR_BINARY_DIR}/mlir-translate -mlir-to-llvmir -o forward193.ll
  COMMAND ${LLVM_MLIR_BINARY_DIR}/clang -O3 ${CLANG_FLAGS_LIST} forward193.ll
    -c -save-temps -o ${CMAKE_CURRENT_BINARY_DIR}/forward193.o
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/graph/forward193.mlir
  COMMENT "Building forward193.o "
  VERBATIM)

add_custom_command(
    OUTPUT subgraph193.o
    COMMAND ${LLVM_MLIR_BINARY_DIR}/mlir-opt ${CMAKE_CURRENT_SOURCE_DIR}/graph/subgraph193.mlir 
              -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named),func.func(tosa-to-linalg),func.func(tosa-to-tensor),func.func(tosa-to-arith))" |
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
            ${LLVM_MLIR_BINARY_DIR}/mlir-translate -mlir-to-llvmir -o subgraph193.ll
  COMMAND ${LLVM_MLIR_BINARY_DIR}/clang -O3 ${CLANG_FLAGS_LIST} subgraph193.ll
    -c -save-temps -o ${CMAKE_CURRENT_BINARY_DIR}/subgraph193.o
    DEPENDS 
      ${CMAKE_CURRENT_SOURCE_DIR}/graph/subgraph193.mlir
    COMMENT "Building subgraph193.o "
    VERBATIM)

add_library(SPLITLLAMA193 STATIC forward193.o subgraph193.o)


find_package(Boost REQUIRED COMPONENTS system thread)
include_directories(${Boost_INCLUDE_DIRS})
include_directories(${Websocket_INCLUDE_DIR})

SET_SOURCE_FILES_PROPERTIES(
  template.o
  PROPERTIES
  EXTERNAL_OBJECT true
  GENERATED true)

SET_TARGET_PROPERTIES(
  SPLITLLAMA0
  SPLITLLAMA1
  SPLITLLAMA2
  SPLITLLAMA3
  SPLITLLAMA5
  SPLITLLAMA193
  PROPERTIES
  LINKER_LANGUAGE C)


function(set_ip_ports target)
  foreach(i RANGE 1 9)
    target_compile_definitions(${target} PRIVATE DEVICE${i}="${DEVICE${i}}")
  endforeach()

  foreach(j RANGE 0 13)
    list(GET PORTS ${j} PORT_VAL)
    math(EXPR PORT_IDX "${j}+1")   
    target_compile_definitions(${target} PRIVATE PORT${PORT_IDX}=${PORT_VAL})
  endforeach()
endfunction()


# Generic Function: Generates runners
# Parameters:
#   target_prefix : e.g., buddy-llama-rms-run
#   source_file   
#   runner_type   : Type (RMS / MHA)
#   ids           : List of IDs (e.g., 0 1 2 3)
function(add_llama_runners target_prefix source_file runner_type ids)
  foreach(ID IN LISTS ids)
    set(exe_name ${target_prefix}-${ID})
    add_executable(${exe_name} ${source_file})

    target_compile_definitions(${exe_name} PRIVATE 
      LLAMA_SPLIT_EXAMPLE_PATH="${LLAMA_SPLIT_EXAMPLE_PATH}" 
      LLAMA_EXAMPLE_BUILD_PATH="${LLAMA_EXAMPLE_BUILD_PATH}"
      PART_NUMS=${PART_NUMS}
      PART_ID=${ID}
    )

    set_ip_ports(${exe_name})

    if(runner_type STREQUAL "INPUT")
      target_link_libraries(${exe_name} PRIVATE SPLITLLAMA0)
    elseif(runner_type STREQUAL "OUTPUT")
      target_link_libraries(${exe_name} PRIVATE SPLITLLAMA193)
    elseif(runner_type STREQUAL "RMS")
      target_link_libraries(${exe_name} PRIVATE SPLITLLAMA1)
    elseif(runner_type STREQUAL "MHA")
      target_link_libraries(${exe_name} PRIVATE SPLITLLAMA2)
    elseif(runner_type STREQUAL "ADD")
      target_link_libraries(${exe_name} PRIVATE SPLITLLAMA3)
    elseif(runner_type STREQUAL "MLP")
      target_link_libraries(${exe_name} PRIVATE SPLITLLAMA5)
    endif()

    target_link_directories(${exe_name} PRIVATE ${LLVM_LIBRARY_DIR})

    target_link_libraries(${exe_name} PRIVATE
      ${Boost_LIBRARIES} 
      pthread
      ${CMAKE_THREAD_LIBS_INIT}
      ${OMP_SHARED_DIR}
      ${BUDDY_LIB_DIR}/libStaticMLIRCRunnerUtils.a
    )

    list(APPEND all_targets ${exe_name})
  endforeach()

  add_custom_target(${target_prefix} ALL DEPENDS ${all_targets})
endfunction()

add_llama_runners(buddy-llama-rms-run    llama-rms-mha.cpp   RMS "${RUNNER_IDS}")
add_llama_runners(buddy-llama-rms-0-run  llama-rms-mha-0.cpp RMS "${RUNNER_IDS}")
add_llama_runners(buddy-llama-mha-run    llama-mha.cpp       MHA "${RUNNER_IDS}")
add_llama_runners(buddy-llama-mha-0-run  llama-mha-0.cpp     MHA "${RUNNER_IDS}")
add_llama_runners(buddy-llama-add-run    llama-add-mha.cpp   ADD "${RUNNER_IDS}")
add_llama_runners(buddy-llama-add-0-run  llama-add-mha-0.cpp ADD "${RUNNER_IDS}")
add_llama_runners(buddy-llama-rms-mlp-run llama-rms-mlp.cpp  RMS "${RUNNER_IDS}")
add_llama_runners(buddy-llama-rms-mlp-0-run llama-rms-mlp-0.cpp  RMS "${RUNNER_IDS}")
add_llama_runners(buddy-llama-mlp-run    llama-mlp.cpp       MLP "${RUNNER_IDS}")
add_llama_runners(buddy-llama-mlp-0-run  llama-mlp-0.cpp     MLP "${RUNNER_IDS}")
add_llama_runners(buddy-llama-add-mlp-run llama-add-mlp.cpp     ADD "${RUNNER_IDS}")
add_llama_runners(buddy-llama-add-mlp-0-run llama-add-mlp-0.cpp ADD "${RUNNER_IDS}")

set(INPUT_IDS 0)
set(OUTPUT_IDS 0)

add_llama_runners(buddy-llama-input-run  llama-input.cpp  INPUT  "${INPUT_IDS}")
add_llama_runners(buddy-llama-output-run llama-output.cpp OUTPUT "${OUTPUT_IDS}")

add_custom_target(buddy-llama-all-runs
  DEPENDS
    buddy-llama-rms-run
    buddy-llama-rms-0-run
    buddy-llama-mha-run
    buddy-llama-mha-0-run
    buddy-llama-add-run
    buddy-llama-add-0-run
    buddy-llama-rms-mlp-run
    buddy-llama-rms-mlp-0-run
    buddy-llama-mlp-run
    buddy-llama-mlp-0-run
    buddy-llama-add-mlp-run
    buddy-llama-add-mlp-0-run
    buddy-llama-input-run
    buddy-llama-output-run
)
