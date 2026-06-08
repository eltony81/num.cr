# Sparse Coordinate (COO) Tensor Module

module Num
  class SparseCOOTensor(T)
    getter rows : Tensor(Int32, CPU(Int32))
    getter cols : Tensor(Int32, CPU(Int32))
    getter data : Tensor(T, CPU(T))
    getter shape : Array(Int32)

    def initialize(
      @rows : Tensor(Int32, CPU(Int32)),
      @cols : Tensor(Int32, CPU(Int32)),
      @data : Tensor(T, CPU(T)),
      @shape : Array(Int32)
    )
      if @rows.size != @cols.size || @rows.size != @data.size
        raise Num::Exceptions::ValueError.new("Sparse arrays rows, cols, and data must have the same size")
      end
      if @shape.size != 2
        raise Num::Exceptions::ValueError.new("SparseCOOTensor only supports 2D matrices")
      end
    end

    # Performs sparse matrix-vector multiplication (y = A * x)
    def matmul(vec : Tensor(T, CPU(T))) : Tensor(T, CPU(T))
      if vec.rank != 1 || vec.size != @shape[1]
        raise Num::Exceptions::ValueError.new("Vector size #{vec.size} must match column dimension #{@shape[1]}")
      end

      result = Tensor(T, CPU(T)).zeros([@shape[0]])
      @rows.size.times do |i|
        r_idx = @rows.to_unsafe[i]
        c_idx = @cols.to_unsafe[i]
        val = @data.to_unsafe[i]
        
        result.to_unsafe[r_idx] += val * vec.to_unsafe[c_idx]
      end
      result
    end
  end
end
