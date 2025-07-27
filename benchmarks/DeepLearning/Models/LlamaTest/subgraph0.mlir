#map = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @subgraph0(%arg0: tensor<32000x4096xf32>, %arg1: tensor<1x40xi64>) -> tensor<1x40x4096xf32> {
    %0 = tosa.cast %arg1 : (tensor<1x40xi64>) -> tensor<1x40xi32>
    %1 = tosa.reshape %arg0 {new_shape = array<i64: 1, 32000, 4096>} : (tensor<32000x4096xf32>) -> tensor<1x32000x4096xf32>
    %2 = tosa.gather %1, %0 : (tensor<1x32000x4096xf32>, tensor<1x40xi32>) -> tensor<1x40x4096xf32>
    %3 = tosa.reshape %2 {new_shape = array<i64: 1, 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
    return %3 : tensor<1x40x4096xf32>
  }
}

