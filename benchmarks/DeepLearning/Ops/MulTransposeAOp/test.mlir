func.func @test(%arg0 : memref<?x?x?x?xf32>, %arg1 : memref<?x?x?x?xf32>, %arg2 : memref<?x?x?x?xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %vl_step = arith.constant 32 : index
  %c0_f32 = arith.constant 0.000000e+00 : f32
  %v0 = vector.splat %c0_f32 : vector<32xf32>
  %dim = memref.dim %arg0, %c0 : memref<?x?x?x?xf32>
  %dim_0 = memref.dim %arg0, %c1 : memref<?x?x?x?xf32>
  %dim_1 = memref.dim %arg0, %c2 : memref<?x?x?x?xf32>
  %dim_2 = memref.dim %arg0, %c3 : memref<?x?x?x?xf32>

  // Calculate the upper bound for vectorized processing
  // - Subtract `vl_step` is to avoid overflow at the vectorization tail.
  // - Add 1 to ensure the final loop runs when the workload length 
  //   is divisible by the vector size.
  %dim_2_upbound_tmp = arith.subi %dim_2, %vl_step : index
  %dim_2_upbound = arith.addi %dim_2_upbound_tmp, %c1 : index

  affine.for %arg3 = %c0 to %dim {
    affine.for %arg4 = %c0 to %dim_1 {
      affine.for %arg5 = %c0 to %dim_0 {
        %iter_idx= scf.for %arg6 = %c0 to %dim_2_upbound 
            step %vl_step iter_args(%iter_init = %c0) -> (index){
          %0 = vector.load %arg0[%arg3, %arg5, %arg4, %arg6] : memref<?x?x?x?xf32>, vector<32xf32>
          %1 = vector.load %arg1[%arg3, %c0, %arg5, %arg6] : memref<?x?x?x?xf32>, vector<32xf32>
          %2 = arith.mulf %0, %1 : vector<32xf32>
          vector.store %2, %arg2[%arg3, %arg4, %arg5, %arg6] : memref<?x?x?x?xf32>, vector<32xf32>
          %dim_2_next = arith.addi %arg6, %vl_step : index
          scf.yield %dim_2_next : index
        }
        // Compute the tail size and Process the remaining elements 
        // using masked vector operations.
        %tail_size = arith.subi %dim_2, %iter_idx : index
        %mask = vector.create_mask %tail_size : vector<32xi1>
        %0 = vector.maskedload %arg0[%arg3, %arg5, %arg4, %iter_idx], %mask, %v0 : memref<?x?x?x?xf32>, vector<32xi1>, vector<32xf32> into vector<32xf32>
        %1 = vector.maskedload %arg1[%arg3, %c0, %arg5, %iter_idx], %mask, %v0 : memref<?x?x?x?xf32>, vector<32xi1>, vector<32xf32> into vector<32xf32>
        %2 = arith.mulf %0, %1 : vector<32xf32>
        vector.maskedstore %arg2[%arg3, %arg4, %arg5, %iter_idx], %mask, %2 : memref<?x?x?x?xf32>, vector<32xi1>, vector<32xf32>
      }
    }
  }
  return
}