#-------------------------------------------------------------------------------
# Generate MLIRFIRScalar
#-------------------------------------------------------------------------------

add_custom_command(
  OUTPUT mlir-fir.o
  COMMAND 
    ${BUDDY_MLIR_BUILD_DIR}/bin/buddy-opt 
      ${BUDDY_SOURCE_DIR}/benchmarks/AudioProcessing/Operations/FIROp/MLIRFIR.mlir
      -convert-scf-to-cf
      -llvm-request-c-wrappers
      -convert-arith-to-llvm
      -finalize-memref-to-llvm
      -convert-func-to-llvm
      -reconcile-unrealized-casts |
    ${LLVM_MLIR_BINARY_DIR}/mlir-translate -mlir-to-llvmir |
    ${LLVM_MLIR_BINARY_DIR}/llc 
      -mtriple=${BUDDY_OPT_TRIPLE} 
      -mattr=${BUDDY_OPT_ATTR} 
      -filetype=obj 
      -o ${BUDDY_BINARY_DIR}/../benchmarks/AudioProcessing/Operations/FIROp/mlir-fir.o
  DEPENDS
    ${BUDDY_MLIR_BUILD_DIR}/bin/buddy-opt
    ${LLVM_MLIR_BINARY_DIR}/mlir-translate
    ${LLVM_MLIR_BINARY_DIR}/llc 
)

add_library(MLIRFIRScalar STATIC mlir-fir.o)
set_target_properties(MLIRFIRScalar PROPERTIES LINKER_LANGUAGE CXX)

#-------------------------------------------------------------------------------
# Generate MLIRFIRTiledVectorization
#-------------------------------------------------------------------------------

add_custom_command(
  OUTPUT fir-tile-vectorization.o
  COMMAND 
    ${BUDDY_MLIR_BUILD_DIR}/bin/buddy-opt 
      ${BUDDY_SOURCE_DIR}/benchmarks/AudioProcessing/Operations/FIROp/MLIRFIRTiledVectorization.mlir
      -convert-scf-to-cf
      -convert-vector-to-llvm
      -llvm-request-c-wrappers
      -convert-arith-to-llvm
      -finalize-memref-to-llvm
      -convert-func-to-llvm
      -reconcile-unrealized-casts |
    ${LLVM_MLIR_BINARY_DIR}/mlir-translate -mlir-to-llvmir |
    ${LLVM_MLIR_BINARY_DIR}/llc 
      -mtriple=${BUDDY_OPT_TRIPLE} 
      -mattr=${BUDDY_OPT_ATTR} 
      -filetype=obj 
      -o ${BUDDY_BINARY_DIR}/../benchmarks/AudioProcessing/Operations/FIROp/fir-tile-vectorization.o
  DEPENDS
    ${BUDDY_MLIR_BUILD_DIR}/bin/buddy-opt
    ${LLVM_MLIR_BINARY_DIR}/mlir-translate
    ${LLVM_MLIR_BINARY_DIR}/llc 
)

add_library(MLIRFIRTiledVectorization STATIC fir-tile-vectorization.o)
set_target_properties(MLIRFIRTiledVectorization PROPERTIES LINKER_LANGUAGE CXX)

#-------------------------------------------------------------------------------
# Generate MLIRFIRVectorization
#-------------------------------------------------------------------------------

add_custom_command(
  OUTPUT fir-vectorization.o
  COMMAND 
    ${BUDDY_MLIR_BUILD_DIR}/bin/buddy-opt 
      ${BUDDY_SOURCE_DIR}/benchmarks/AudioProcessing/Operations/FIROp/MLIRFIRVectorization.mlir
      -convert-scf-to-cf
      -convert-vector-to-llvm
      -llvm-request-c-wrappers
      -convert-arith-to-llvm
      -finalize-memref-to-llvm
      -convert-func-to-llvm
      -reconcile-unrealized-casts |
    ${LLVM_MLIR_BINARY_DIR}/mlir-translate -mlir-to-llvmir |
    ${LLVM_MLIR_BINARY_DIR}/llc 
      -mtriple=${BUDDY_OPT_TRIPLE} 
      -mattr=${BUDDY_OPT_ATTR} 
      -filetype=obj 
      -o ${BUDDY_BINARY_DIR}/../benchmarks/AudioProcessing/Operations/FIROp/fir-vectorization.o
  DEPENDS
    ${BUDDY_MLIR_BUILD_DIR}/bin/buddy-opt
    ${LLVM_MLIR_BINARY_DIR}/mlir-translate
    ${LLVM_MLIR_BINARY_DIR}/llc 
)

add_library(MLIRFIRVectorization STATIC fir-vectorization.o)
set_target_properties(MLIRFIRVectorization PROPERTIES LINKER_LANGUAGE CXX)

#-------------------------------------------------------------------------------
# Generate dap-op-fir-benchmark
#-------------------------------------------------------------------------------

add_executable(dap-op-fir-benchmark Main.cpp)

target_link_directories(dap-op-fir-benchmark PRIVATE 
  ${KFR_DIR}/build/
  ${BUDDY_MLIR_LIB_DIR}
)

target_link_libraries(dap-op-fir-benchmark PRIVATE
  # Third-party library
  kfr_io
  # MLIR hand-written benchmark
  MLIRFIRScalar
  MLIRFIRTiledVectorization
  MLIRFIRVectorization
  # Buddy DAP library
  BuddyLibDAP
  # LLVM/MLIR library
  StaticMLIRCRunnerUtils
  # Benchmark library
  GoogleBenchmark
)
