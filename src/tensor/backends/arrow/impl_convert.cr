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

module Num
  # Converts a `Tensor` to a standard library array.  The returned array
  # will always be one-dimensional to avoid return type ambiguity
  #
  # ## Arguments
  #
  # * arr : `Tensor(U, ARROW(U))` - `Tensor` to convert to an `Array`
  #
  # ## Examples
  #
  # ```
  # a = Tensor.new([2, 2]) { |i| i }
  # a.to_a # => [0, 1, 2, 3]
  # ```
  def to_a(arr : Tensor(U, ARROW(U))) forall U
    a = [] of U
    each(arr) do |el|
      a << el
    end
    a
  end

  # Places a `Tensor` stored on a ARROW onto an OpenCL Device.
  #
  # ## Arguments
  #
  # * arr : `Tensor(U, ARROW(U))` - `Tensor` to place on OpenCL device.
  #
  # ## Examples
  #
  # ```
  # a = Tensor.new([2, 2]) { |i| i }
  # a.opencl # => "<4> on OpenCL Backend"
  # ```
  def opencl(arr : Tensor(U, ARROW(U))) : Tensor(U, OCL(U)) forall U
    unless arr.flags.contiguous?
      arr = arr.dup
    end
    storage = OCL(U).new(arr.to_unsafe, arr.shape, arr.strides)
    Tensor(U, OCL(U)).new(storage, arr.shape)
  end

  # Casts a `Tensor` to a new dtype, by making a copy.  Information may
  # be lost when converting between data types, for example Float to Int
  # or Int to Bool.
  #
  # ## Arguments
  #
  # * u : `U.class` - Data type the `Tensor` will be cast to
  #
  # ## Examples
  #
  # ```
  # a = Tensor.from_array [1.5, 2.5, 3.5]
  #
  # a.astype(Int32)   # => [1, 2, 3]
  # a.astype(Bool)    # => [true, true, true]
  # a.astype(Float32) # => [1.5, 2.5, 3.5]
  # ```
  def as_type(arr : Tensor(U, ARROW(U)), dtype : V.class) forall U, V
    r = Tensor(V, ARROW(V)).new(arr.shape)
    r.map!(arr) do |_, j|
      {% if U == Bool %}
        j ? 1 : 0
      {% else %}
        j
      {% end %}
    end
    r
  end

  # Converts a ARROW `Tensor` to CPU.  Returns the input array,
  # no copy is performed.
  #
  # ## Arguments
  #
  # * arr : `Tensor(U, ARROW(U))` - `Tensor` to return
  #
  # ## Examples
  #
  # ```
  # a = Tensor.new([2, 2]) { |i| i }
  # a.opencl # => "<4> on OpenCL Backend"
  # ```
  def cpu(arr : Tensor(U, ARROW(U))) forall U
    storage = CPU(U).new(arr.to_unsafe, arr.shape, arr.strides)
    Tensor(U, CPU(U)).new(storage, arr.shape)
  end

  # Exports a `Tensor` to an `Arrow::Tensor` (multidimensional array)
  def to_arrow_tensor(arr : Tensor(U, ARROW(U))) : Arrow::Tensor forall U
    dtype = case U
            when Int8.class    then Arrow::DataType.int8
            when UInt8.class   then Arrow::DataType.uint8
            when Int16.class   then Arrow::DataType.int16
            when UInt16.class  then Arrow::DataType.uint16
            when Int32.class   then Arrow::DataType.int32
            when UInt32.class  then Arrow::DataType.uint32
            when Int64.class   then Arrow::DataType.int64
            when UInt64.class  then Arrow::DataType.uint64
            when Float32.class then Arrow::DataType.float
            when Float64.class then Arrow::DataType.double
            when Bool.class    then Arrow::DataType.boolean
            else
              raise "Unsupported data type for Arrow::Tensor export"
            end

    # Zero-copy wrap of the raw pointer in GArrowBuffer
    raw_ptr = arr.to_unsafe
    size_in_bytes = arr.shape.product * sizeof(U)
    bytes = Bytes.new(raw_ptr.unsafe_as(Pointer(UInt8)), size_in_bytes)
    buffer = Arrow::Buffer.new(bytes)

    # GArrowTensor shape and byte strides
    shape_i64 = arr.shape.map(&.to_i64)
    strides_i64 = arr.strides.map { |s| (s * sizeof(U)).to_i64 }

    Arrow::Tensor.new(dtype, buffer, shape_i64, strides_i64)
  end

  # Creates a `Tensor` from an `Arrow::Tensor` (multidimensional array)
  def from_arrow_tensor(tensor : Arrow::Tensor, dtype : U.class) : Tensor(U, ARROW(U)) forall U
    # Get the raw memory pointer from GArrowBuffer (zero-copy)
    buffer = tensor.buffer
    raw_ptr = buffer.raw_pointer.as(Pointer(U))

    # Convert GArrowTensor shape and byte strides back to elements
    shape = tensor.shape.map(&.to_i32)
    strides = tensor.strides.map { |s| (s / sizeof(U)).to_i32 }

    storage = ARROW(U).new(raw_ptr, shape, strides)
    Tensor(U, ARROW(U)).new(storage, shape, strides, 0)
  end
end
