func.func @kernel_placeholder(%a : tensor<?x?x?xf32>, %b : tensor<?x?x?xf32>) -> (tensor<?x?x?xf32>) {
    %0 = "tosa.const"() <{value = dense<[1, 0, 2]> : tensor<3xi32>}> : () -> tensor<3xi32>
    %1 = tosa.transpose %b, %0 : (tensor<?x?x?xf32>, tensor<3xi32>) -> tensor<?x?x?xf32>
    %2 = tosa.matmul %a, %1 : (tensor<?x?x?xf32>, tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
    %3 = "tosa.const"() <{value = dense<[1, 0, 2]> : tensor<3xi32>}> : () -> tensor<3xi32>
    %c = tosa.transpose %2, %3 : (tensor<?x?x?xf32>, tensor<3xi32>) -> tensor<?x?x?xf32>  
    return %c : tensor<?x?x?xf32>
  }
