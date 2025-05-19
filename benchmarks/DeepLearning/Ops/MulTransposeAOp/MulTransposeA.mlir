func.func @mul_transpose_a(%arg0 : tensor<1x40x32x128xf32>, %arg1 : tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32> {
  %0 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
  %1 = tosa.transpose %arg0, %0 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
  %2 = tosa.mul %1, %arg1 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>

  return %2 : tensor<1x32x40x128xf32>
}
