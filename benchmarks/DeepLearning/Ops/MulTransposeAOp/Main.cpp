//===- Main.cpp -----------------------------------------------------------===//
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//===----------------------------------------------------------------------===//
//
// This is the main file of the Conv2D NHWC-FHWC benchmark.
//
//===----------------------------------------------------------------------===//

#include "Utils.hpp"
#include <benchmark/benchmark.h>
#include <buddy/Core/Container.h>
#include <iostream>

// -----------------------------------------------------------------------------
// Benchmark Configuration. You can change the numbers here as needed.
// -----------------------------------------------------------------------------

#define NUM_ITER 5
#define INPUT_N 1
#define INPUT_C 40
#define INPUT_H 32
#define INPUT_W 128
#define FILTER_F INPUT_N
#define FILTER_C 1
#define FILTER_H INPUT_C
#define FILTER_W INPUT_W
#define OUTPUT_N INPUT_N
#define OUTPUT_C INPUT_H
#define OUTPUT_H INPUT_C
#define OUTPUT_W INPUT_W


// -----------------------------------------------------------------------------
// Global Variables and Functions. No need to change the code here.
// -----------------------------------------------------------------------------

intptr_t sizesInput[4] = {INPUT_N, INPUT_C, INPUT_H, INPUT_W};
intptr_t sizesFilter[4] = {FILTER_F, FILTER_C, FILTER_H, FILTER_W};
intptr_t sizesOutput[4] = {OUTPUT_N, OUTPUT_C, OUTPUT_H, OUTPUT_W};

float *inputData = nullptr;
float *filterData = nullptr;

MemRef<float, 4> inputMemRef(sizesInput);
MemRef<float, 4> filterMemRef(sizesFilter);

// Runs the provided Conv2D function for benchmarking.
template <typename Func>
void DL_OPS_MUL_TRANSPOSE_A(benchmark::State &state, Func func) {
  MemRef<float, 4> outputMemRef(sizesOutput, 0.0);
  for (auto _ : state) {
    func(&inputMemRef, &filterMemRef, &outputMemRef);
  }
  benchmark::DoNotOptimize(outputMemRef);
}

using MLIRFunctionType = void (*)(MemRef<float, 4> *, MemRef<float, 4> *,
                                  MemRef<float, 4> *);

// Verifies the result of an MLIR-based function against expected output.
void MLIRVerification(float *outputExpected, MLIRFunctionType MLIRFunc,
                      const std::string &name) {
  MemRef<float, 4> outputMemRef(sizesOutput, 0);
  MLIRFunc(&inputMemRef, &filterMemRef, &outputMemRef);
  float *outputOptimized = outputMemRef.getData();
  mul_transpose_a::verify<float>(outputExpected, outputOptimized,
                               OUTPUT_N * OUTPUT_C * OUTPUT_H * OUTPUT_W, name);
}

// -----------------------------------------------------------------------------
// MLIR Benchmark. You can compare your new method with other methods here.
// -----------------------------------------------------------------------------

extern "C" {
void _mlir_ciface_mul_transpose_a_scalar_O0(MemRef<float, 4> *input,
                                            MemRef<float, 4> *filter,
                                            MemRef<float, 4> *output);
void _mlir_ciface_mul_transpose_a_scalar_O3(MemRef<float, 4> *input,
                                            MemRef<float, 4> *filter,
                                            MemRef<float, 4> *output);
void _mlir_ciface_mul_transpose_a_vectorization(MemRef<float, 4> *input,
                                                MemRef<float, 4> *filter,
                                                MemRef<float, 4> *output);
/// [Step 1] Add function of your new method here.
}

BENCHMARK_CAPTURE(DL_OPS_MUL_TRANSPOSE_A, scalar_O0,
                  _mlir_ciface_mul_transpose_a_scalar_O0)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(NUM_ITER);
BENCHMARK_CAPTURE(DL_OPS_MUL_TRANSPOSE_A, scalar_O3,
                  _mlir_ciface_mul_transpose_a_scalar_O3)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(NUM_ITER);
BENCHMARK_CAPTURE(DL_OPS_MUL_TRANSPOSE_A, vectorization,
                  _mlir_ciface_mul_transpose_a_vectorization)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(NUM_ITER);
/// [Step 2] Call GoogleBenchmark function to run your new method.

// -----------------------------------------------------------------------------
// Main Function. You can verify the correctness of your new method here.
// -----------------------------------------------------------------------------

int main(int argc, char **argv) {
  // Initialize input data.
  inputData =
      mul_transpose_a::allocArray<float>(INPUT_N, INPUT_C, INPUT_H, INPUT_W);
  filterData =
      mul_transpose_a::allocArray<float>(FILTER_F, FILTER_C, FILTER_H, FILTER_W);
  // float *input = inputMemRef.getData();
  inputMemRef = MemRef<float, 4>(inputData, sizesInput);
  filterMemRef = MemRef<float, 4>(filterData, sizesFilter);

  // Run benchmark.
  ::benchmark::Initialize(&argc, argv);
  ::benchmark::RunSpecifiedBenchmarks();

  std::cout << "\033[34m---------- Verification ----------\033[0m" << std::endl;
  // Obtain scalar output results as expected output for verification.
  MemRef<float, 4> outputMemRefScalar(sizesOutput, 0);
  _mlir_ciface_mul_transpose_a_scalar_O0(&inputMemRef, &filterMemRef,
                                      &outputMemRefScalar);
  float *outputExpected = outputMemRefScalar.getData();

  MLIRVerification(outputExpected, _mlir_ciface_mul_transpose_a_scalar_O3, "O3");
  MLIRVerification(outputExpected, _mlir_ciface_mul_transpose_a_vectorization,
                   "vectorization");
  /// [Step 3] Add your new method for verification.

  delete[] inputData;
  delete[] filterData;
  return 0;
}
