//===- Utils.hpp ----------------------------------------------------------===//
//
// Licensed under the Apache License, Version 2.0 (the "License");
// ...

#ifndef MUL_TRANSPOSE_A_UTILS_HPP
#define MUL_TRANSPOSE_A_UTILS_HPP

#include <benchmark/benchmark.h>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <iostream>

// -----------------------------------------------------------------------------
// Helper Functions
// -----------------------------------------------------------------------------

namespace mul_transpose_a {

// Allocates a 1D array with dimensions `dim0 * dim1 * dim2 * dim3` and fills it
// with random integer values between -500 and 500.
template <typename DATA_TYPE>
DATA_TYPE *allocArray(int dim0, int dim1, int dim2, int dim3) {
  // Initialize the random number generator.
  std::srand(static_cast<unsigned int>(std::time(0)));
  // Calculate the total size.
  int size = dim0 * dim1 * dim2 * dim3;
  // Allocate memory for the array.
  DATA_TYPE *array = new DATA_TYPE[size];
  // Fill the array with random numbers between -500 and 500.
  for (int i = 0; i < size; ++i) {
    array[i] = static_cast<DATA_TYPE>(std::rand() % 100);
  }
  return array;
}

template <typename DATA_TYPE>
void verify(DATA_TYPE *A, DATA_TYPE *B, int size, const std::string &name) {
  const std::string PASS = "\033[32mPASS\033[0m";
  const std::string FAIL = "\033[31mFAIL\033[0m";
  const double epsilon = 1e-2; // Tolerance for floating point comparison.

  std::cout << name << " ";
  if (!A || !B) {
    std::cout << FAIL << " (Null pointer detected)" << std::endl;
    return;
  }

  bool isPass = true;
  for (int i = 0; i < size; ++i) {
    if (std::fabs((A[i] - B[i]) / (B[i] == 0 ? 1 : B[i])) > epsilon) {
      std::cout << FAIL << std::endl;
      std::cout << "Index " << i << ":\tA=" << A[i] << " B=" << B[i]
                << std::endl;
      isPass = false;
      break;
    }
  }
  if (isPass) {
    std::cout << PASS << std::endl;
  }
}

} // namespace mul_transpose_a

#endif // MUL_TRANSPOSE_A_UTILS_HPP
