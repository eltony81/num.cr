# Masked Tensors Module

module Num
  class MaskedTensor(T, S, M)
    getter data : Tensor(T, S)
    getter mask : Tensor(Bool, M) # true indicates masked (invalid/ignored)

    def initialize(@data : Tensor(T, S), @mask : Tensor(Bool, M))
      if @data.shape != @mask.shape
        raise Num::Exceptions::ValueError.new("Mask shape #{@mask.shape} must match data shape #{@data.shape}")
      end
    end

    def shape
      @data.shape
    end

    def size
      @data.size
    end

    # Computes sum of only unmasked elements
    def sum : T
      total = T.zero
      @data.zip(@mask) do |val, is_masked|
        unless is_masked
          total += val
        end
      end
      total
    end

    # Computes mean of only unmasked elements
    def mean : Float64
      total = 0.0
      count = 0
      @data.zip(@mask) do |val, is_masked|
        unless is_masked
          total += val.to_f64
          count += 1
        end
      end
      count == 0 ? 0.0 : total / count
    end

    # Retrieves all unmasked elements as a flat array
    def to_a : Array(T)
      arr = [] of T
      @data.zip(@mask) do |val, is_masked|
        unless is_masked
          arr << val
        end
      end
      arr
    end

    # Element-wise addition combining masks (propagates true mask)
    def +(other : MaskedTensor(T, S, M)) : MaskedTensor(T, S, M)
      new_data = @data + other.data
      new_mask = @mask | other.mask
      MaskedTensor(T, S, M).new(new_data, new_mask)
    end

    # Element-wise subtraction combining masks
    def -(other : MaskedTensor(T, S, M)) : MaskedTensor(T, S, M)
      new_data = @data - other.data
      new_mask = @mask | other.mask
      MaskedTensor(T, S, M).new(new_data, new_mask)
    end

    # Element-wise multiplication combining masks
    def *(other : MaskedTensor(T, S, M)) : MaskedTensor(T, S, M)
      new_data = @data * other.data
      new_mask = @mask | other.mask
      MaskedTensor(T, S, M).new(new_data, new_mask)
    end
  end
end
