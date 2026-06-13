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
  # * arr : `Tensor(U, OCL(U))` - `Tensor` to convert to a stdlib array
  #
  # ## Examples
  #
  # ```
  # a = Tensor.new([2, 2], device: OCL) { |i| i }
  # a.to_a # => [0, 1, 2, 3]
  # ```
  def to_a(arr : Tensor(U, OCL(U))) forall U
    a = Array(U).new(arr.size, 0)
    storage = arr.data
    data_field = storage.data
    if data_field.is_a?(Cl::SVMPointer)
      Cl.map_svm(Num::ClContext.instance.queue, LibCL::CL_TRUE, LibCL::ClMapFlags::READ.value, data_field.raw, (storage.total_size * sizeof(U)).to_u64)
      a.to_unsafe.copy_from(data_field.raw.as(U*) + arr.offset, arr.size)
      Cl.unmap_svm(Num::ClContext.instance.queue, data_field.raw)
    else
      LibCL.cl_enqueue_read_buffer(
        Num::ClContext.instance.queue,
        data_field,
        LibCL::CL_TRUE,
        0_u64,
        (storage.total_size * sizeof(U)).to_u64,
        a.to_unsafe,
        0_u32,
        nil,
        nil
      )
    end
    a
  end

  # Converts a `Tensor` stored on an OpenCL device to a `Tensor` stored
  # on a CPU.
  #
  # ## Arguments
  #
  # * arr : `Tensor(U, OCL(U))` - `Tensor` to place on a CPU
  #
  # ## Examples
  #
  # ```
  # a = Tensor.new([2, 2], device: OCL) { |i| i }
  # a.cpu # => [[0, 1], [2, 3]]
  # ```
  def cpu(arr : Tensor(U, OCL(U))) forall U
    ptr = Pointer(U).malloc(arr.data.total_size)
    storage = arr.data
    data_field = storage.data
    if data_field.is_a?(Cl::SVMPointer)
      Cl.map_svm(Num::ClContext.instance.queue, LibCL::CL_TRUE, LibCL::ClMapFlags::READ.value, data_field.raw, (storage.total_size * sizeof(U)).to_u64)
      ptr.copy_from(data_field.raw.as(U*), storage.total_size)
      Cl.unmap_svm(Num::ClContext.instance.queue, data_field.raw)
    else
      LibCL.cl_enqueue_read_buffer(
        Num::ClContext.instance.queue,
        data_field,
        LibCL::CL_TRUE,
        0_u64,
        (storage.total_size * sizeof(U)).to_u64,
        ptr,
        0_u32,
        nil,
        nil
      )
    end
    Tensor(U, CPU(U)).new(
      CPU(U).new(ptr, arr.shape, arr.strides),
      arr.shape,
      arr.strides,
      arr.offset,
      U
    )
  end
end
