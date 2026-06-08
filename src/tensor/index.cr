# Copyright (c) 2020 Crystal Data Contributors
#
# MIT License
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to
# the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
# LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
# WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

class Tensor(T, S)
  # Returns a view of a `Tensor` from any valid indexers. This view
  # must be able to be represented as valid strided/shaped view, slicing
  # as a copy is not supported.
  #
  #
  # When an Integer argument is passed, an axis will be removed from
  # the `Tensor`, and a view at that index will be returned.
  #
  # ```
  # a = Tensor.new([2, 2]) { |i| i }
  # a[0] # => [0, 1]
  # ```
  #
  # When a Range argument is passed, an axis will be sliced based on
  # the endpoints of the range.
  #
  # ```
  # a = Tensor.new([2, 2, 2]) { |i| i }
  # a[1...]
  #
  # # [[[4, 5],
  # #   [6, 7]]]
  # ```
  #
  # When a Tuple containing a Range and an Integer step is passed, an axis is
  # sliced based on the endpoints of the range, and the strides of the
  # axis are updated to reflect the step.  Negative steps will reflect
  # the array along an axis.
  #
  # ```
  # a = Tensor.new([2, 2]) { |i| i }
  # a[{..., -1}]
  #
  # # [[2, 3],
  # #  [0, 1]]
  # ```
  def [](*args) : Tensor(T, S)
    slice(args.to_a)
  end

  # Returns a view of a `Tensor` from any valid indexers. This view
  # must be able to be represented as valid strided/shaped view, slicing
  # as a copy is not supported.
  #
  #
  # When an Integer argument is passed, an axis will be removed from
  # the `Tensor`, and a view at that index will be returned.
  #
  # ```
  # a = Tensor.new([2, 2]) { |i| i }
  # a[0] # => [0, 1]
  # ```
  #
  # When a Range argument is passed, an axis will be sliced based on
  # the endpoints of the range.
  #
  # ```
  # a = Tensor.new([2, 2, 2]) { |i| i }
  # a[1...]
  #
  # # [[[4, 5],
  # #   [6, 7]]]
  # ```
  #
  # When a Tuple containing a Range and an Integer step is passed, an axis is
  # sliced based on the endpoints of the range, and the strides of the
  # axis are updated to reflect the step.  Negative steps will reflect
  # the array along an axis.
  #
  # ```
  # a = Tensor.new([2, 2]) { |i| i }
  # a[{..., -1}]
  #
  # # [[2, 3],
  # #  [0, 1]]
  # ```
  def [](args : Array) : Tensor(T, S)
    slice(args)
  end

  # An indexing operation that accepts another Tensor for advanced indexing capabilities.
  #
  # This method allows you to index a tensor by using another tensor as indices.
  # It's particularly useful when you need to select or change complex patterns that can't be achieved with simple slicing.
  #
  # The shape of the returned tensor is determined by the shape of the `indices` tensor concatenated with the remaining dimensions of the original tensor.
  # Each value in `indices` selects a value from the original tensor at the corresponding index.
  # If the corresponding index is a vector, than the returned tensor will include the vector as an additional dimension.
  #
  # ## Parameters:
  #
  # * `indices` - A Tensor that contains the indices to index the original tensor.
  #
  # ## Returns:
  #
  # A new tensor containing the values of the original tensor indexed by `indices`.
  #
  # ## Examples:
  #
  # ```
  # t = Tensor.new([[1, 2], [3, 4], [5, 6]])
  # indices = Tensor.new([[0, 2]])
  # result = t[indices] # Returns: Tensor([[1, 2], [5, 6]])
  # ```
  #
  def [](indices : Tensor(Bool, S2)) : Tensor(T, S) forall S2
    # Boolean indexing / masking
    if indices.shape != self.shape
      raise Num::Exceptions::ValueError.new("Boolean mask shape #{indices.shape} must match tensor shape #{self.shape}")
    end
    count = 0
    indices.each do |val|
      count += 1 if val
    end
    result = Tensor(T, S).zeros([count])
    idx = 0
    self.zip(indices) do |val, m_val|
      if m_val
        result.to_unsafe[idx] = val
        idx += 1
      end
    end
    result
  end

  def [](indices : Tensor(U, S2)) : Tensor(T, S) forall U, S2
    # Integer array indexing
    shape = indices.shape + self.shape[1..]
    o = Tensor(T, S).zeros(shape)
    indices.shape[0].times do |i|
      row = indices[i]
      if row.is_a?(Tensor) && row.rank == 0 && row.size == 1
        o[i] = self[row.value.to_i]
      else
        o[i] = self[row]
      end
    end
    o
  end

  def []=(mask : Tensor(Bool, S2), value : T) forall S2
    unless @flags.write?
      raise Num::Exceptions::ValueError.new("Tensor is read-only")
    end
    if mask.shape != self.shape
      raise Num::Exceptions::ValueError.new("Boolean mask shape #{mask.shape} must match tensor shape #{self.shape}")
    end
    self.map!(mask) do |val, mask_val|
      mask_val ? value : val
    end
  end

  def []=(mask : Tensor(Bool, S2), value : Tensor(T, S3)) forall S2, S3
    unless @flags.write?
      raise Num::Exceptions::ValueError.new("Tensor is read-only")
    end
    if mask.shape != self.shape
      raise Num::Exceptions::ValueError.new("Boolean mask shape #{mask.shape} must match tensor shape #{self.shape}")
    end
    val_idx = 0
    self.map!(mask) do |val, mask_val|
      if mask_val
        ret = value.to_unsafe[val_idx]
        val_idx += 1
        ret
      else
        val
      end
    end
  end






  # The primary method of setting Tensor values.  The slicing behavior
  # for this method is identical to the `[]` method.
  #
  # If a `Tensor` is passed as the value to set, it will be broadcast
  # to the shape of the slice if possible.  If a scalar is passed, it will
  # be tiled across the slice.
  #
  # ## Arguments
  #
  # * args : `*U` - Tuple of arguments.  All but the last argument must be
  #   valid indexer, so a `Range`, `Int`, or `Tuple(Range, Int)`.  The final
  #   argument passed is used to set the values of the `Tensor`.  It can
  #   be either a `Tensor`, or a scalar value.
  #
  # ## Examples
  #
  # ```
  # a = Tensor.new([2, 2]) { |i| i }
  # a[1.., 1..] = 99
  # a
  #
  # # [[ 0,  1],
  # #  [ 2, 99]]
  # ```
  def []=(*args : *U) forall U
    unless @flags.write?
      raise Num::Exceptions::ValueError.new("Tensor is read-only")
    end
    {% begin %}
       set(
         {% for i in 0...U.size - 1 %}
           args[{{i}}],
         {% end %}
         value: args[{{U.size - 1}}]
       )
     {% end %}
  end

  # The primary method of setting Tensor values.  The slicing behavior
  # for this method is identical to the `[]` method.
  #
  # If a `Tensor` is passed as the value to set, it will be broadcast
  # to the shape of the slice if possible.  If a scalar is passed, it will
  # be tiled across the slice.
  #
  # ## Arguments
  #
  # * args : `Array` - Array of arguments.  All but the last argument must be
  #   valid indexer, so a `Range`, `Int`, or `Tuple(Range, Int)`.
  # * value : `Tensor` | `Number` - Value to assign to the slice
  #
  # ## Examples
  #
  # ```
  # a = Tensor.new([2, 2]) { |i| i }
  # a[1.., 1..] = 99
  # a
  #
  # # [[ 0,  1],
  # #  [ 2, 99]]
  # ```
  def []=(args : Array, value)
    unless @flags.write?
      raise Num::Exceptions::ValueError.new("Tensor is read-only")
    end
    Num.set(self, args, value)
  end

  # :nodoc:
  def value : T
    @data.to_hostptr[offset]
  end

  # Returns a view of a `Tensor` from any valid indexers. This view
  # must be able to be represented as valid strided/shaped view, slicing
  # as a copy is not supported.
  #
  #
  # When an Integer argument is passed, an axis will be removed from
  # the `Tensor`, and a view at that index will be returned.
  #
  # ```
  # a = Tensor.new([2, 2]) { |i| i }
  # a[0] # => [0, 1]
  # ```
  #
  # When a Range argument is passed, an axis will be sliced based on
  # the endpoints of the range.
  #
  # ```
  # a = Tensor.new([2, 2, 2]) { |i| i }
  # a[1...]
  #
  # # [[[4, 5],
  # #   [6, 7]]]
  # ```
  #
  # When a Tuple containing a Range and an Integer step is passed, an axis is
  # sliced based on the endpoints of the range, and the strides of the
  # axis are updated to reflect the step.  Negative steps will reflect
  # the array along an axis.
  #
  # ```
  # a = Tensor.new([2, 2]) { |i| i }
  # a[{..., -1}]
  #
  # # [[2, 3],
  # #  [0, 1]]
  # ```
  def slice(*args) : Tensor(T, S)
    Num.slice(self, *args)
  end

  # The primary method of setting Tensor values.  The slicing behavior
  # for this method is identical to the `[]` method.
  #
  # If a `Tensor` is passed as the value to set, it will be broadcast
  # to the shape of the slice if possible.  If a scalar is passed, it will
  # be tiled across the slice.
  #
  # ## Arguments
  #
  # * args : `Tuple` - Tuple of arguments.  All must be
  #   valid indexers, so a `Range`, `Int`, or `Tuple(Range, Int)`.
  # * value : `Tensor | Number` - Value to assign to the slice
  #
  # ## Examples
  #
  # ```
  # a = Tensor.new([2, 2]) { |i| i }
  # a[1.., 1..] = 99
  # a
  #
  # # [[ 0,  1],
  # #  [ 2, 99]]
  # ```
  def set(*args, value)
    unless @flags.write?
      raise Num::Exceptions::ValueError.new("Tensor is read-only")
    end
    Num.set(self, *args, value: value)
  end

  # Return a shallow copy of a `Tensor` with a new dtype.  The underlying
  # data buffer is shared, but the `Tensor` owns its other attributes.
  # The size of the new dtype must be a multiple of the current dtype
  #
  # ## Arguments
  #
  # * u : `U.class` - The data type used to reintepret the underlying data buffer
  #   of a `Tensor`
  #
  # ## Examples
  #
  # ```
  # a = Tensor.new([3]) { |i| i }
  # a.view(Int8) # => [0, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0]
  # ```
  def view(u : U.class) forall U
    Num.view(self, U)
  end
end
