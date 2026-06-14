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

class OCL(T) < Num::Backend::Storage(T)
  # Initialize an OpenCL storage from an initial capacity.
  # The data will be filled with zeros
  #
  # ## Arguments
  #
  # * shape : `Array(Int)` - Shape for parent `Tensor`
  # * order : `Num::OrderType` - Memory layout for parent `Tensor`
  #
  # ## Examples
  #
  # ```
  # OCL.new([100], Num::RowMajor)
  # ```
  private def allocate_buffer(size : UInt64, dtype : U.class) : LibCL::ClMem | Cl::SVMPointer forall U
    if Num::ClContext.instance.svm_supported?
      flags = 1_u64 << 0 # CL_MEM_READ_WRITE
      if Num::ClContext.instance.fine_grain_svm_supported?
        flags |= 1_u64 << 10 # CL_MEM_SVM_FINE_GRAIN_BUFFER
      end
      ptr = LibCL.cl_svm_alloc(Num::ClContext.instance.context, flags, size * sizeof(U), 0_u32)
      if ptr.nil?
        raise "Failed to allocate SVM pointer"
      end
      Cl::SVMPointer.new(ptr)
    else
      Cl.buffer(Num::ClContext.instance.context, size, dtype: U)
    end
  end

  def initialize(@data : LibCL::ClMem | Cl::SVMPointer, shape : Array(Int), strides : Array(Int))
    @shape = metadata_to_buffer(shape.map &.to_i)
    @strides = metadata_to_buffer(strides.map &.to_i)
    @total_size = shape.product
  end

  def initialize(shape : Array(Int), order : Num::OrderType)
    @data = allocate_buffer(shape.product.to_u64, T)
    @shape = metadata_to_buffer(shape.map &.to_i)
    @strides = metadata_to_buffer(Num::Internal.shape_to_strides(shape, order))
    @total_size = shape.product
  end

  def initialize(shape : Array(Int), strides : Array(Int))
    @data = allocate_buffer(shape.product.to_u64, T)
    @shape = metadata_to_buffer(shape.map &.to_i)
    @strides = metadata_to_buffer(strides.map &.to_i)
    @total_size = shape.product
  end

  def initialize(shape : Array(Int), order : Num::OrderType, value : T)
    @data = allocate_buffer(shape.product.to_u64, T)
    @shape = metadata_to_buffer(shape.map &.to_i)
    @strides = metadata_to_buffer(Num::Internal.shape_to_strides(shape, order))
    @total_size = shape.product
    if (data_ptr = @data).is_a?(Cl::SVMPointer)
      unless Num::ClContext.instance.fine_grain_svm_supported?
        Cl.map_svm(Num::ClContext.instance.queue, LibCL::CL_TRUE, LibCL::ClMapFlags::WRITE.value, data_ptr.raw, (shape.product * sizeof(T)).to_u64)
      end
      data_ptr.raw.as(T*).map!(shape.product) { value }
      unless Num::ClContext.instance.fine_grain_svm_supported?
        Cl.unmap_svm(Num::ClContext.instance.queue, data_ptr.raw)
      end
    else
      Cl.fill(Num::ClContext.instance.queue, data_ptr, value, shape.product.to_u64)
    end
  end

  def initialize(shape : Array(Int), strides : Array(Int), value : T)
    @data = allocate_buffer(shape.product.to_u64, T)
    @shape = metadata_to_buffer(shape.map &.to_i)
    @strides = metadata_to_buffer(strides.map &.to_i)
    @total_size = shape.product
    if (data_ptr = @data).is_a?(Cl::SVMPointer)
      unless Num::ClContext.instance.fine_grain_svm_supported?
        Cl.map_svm(Num::ClContext.instance.queue, LibCL::CL_TRUE, LibCL::ClMapFlags::WRITE.value, data_ptr.raw, (shape.product * sizeof(T)).to_u64)
      end
      data_ptr.raw.as(T*).map!(shape.product) { value }
      unless Num::ClContext.instance.fine_grain_svm_supported?
        Cl.unmap_svm(Num::ClContext.instance.queue, data_ptr.raw)
      end
    else
      Cl.fill(Num::ClContext.instance.queue, data_ptr, value, shape.product.to_u64)
    end
  end

  def initialize(hostptr : Pointer(T), shape : Array(Int), strides : Array(Int))
    @data = allocate_buffer(shape.product.to_u64, T)
    @shape = metadata_to_buffer(shape.map &.to_i)
    @strides = metadata_to_buffer(strides.map &.to_i)
    @total_size = shape.product
    if (data_ptr = @data).is_a?(Cl::SVMPointer)
      unless Num::ClContext.instance.fine_grain_svm_supported?
        Cl.map_svm(Num::ClContext.instance.queue, LibCL::CL_TRUE, LibCL::ClMapFlags::WRITE.value, data_ptr.raw, (shape.product * sizeof(T)).to_u64)
      end
      data_ptr.raw.as(T*).copy_from(hostptr, shape.product)
      unless Num::ClContext.instance.fine_grain_svm_supported?
        Cl.unmap_svm(Num::ClContext.instance.queue, data_ptr.raw)
      end
    else
      Cl.write(Num::ClContext.instance.queue, hostptr, data_ptr, (shape.product * sizeof(T)).to_u64)
    end
  end

  def update_metadata(shape : Array(Int32), strides : Array(Int32))
    free_buffer(@shape)
    free_buffer(@strides)
    @shape = metadata_to_buffer(shape)
    @strides = metadata_to_buffer(strides)
  end

  def self.base(dtype : U.class) : OCL(U).class forall U
    OCL(U)
  end

  private def metadata_to_buffer(arr : Array(Int32)) : LibCL::ClMem | Cl::SVMPointer
    if arr == [] of Int32
      arr = [1]
    end
    buf = allocate_buffer(arr.size.to_u64, Int32)
    if buf.is_a?(Cl::SVMPointer)
      unless Num::ClContext.instance.fine_grain_svm_supported?
        Cl.map_svm(Num::ClContext.instance.queue, LibCL::CL_TRUE, LibCL::ClMapFlags::WRITE.value, buf.raw, (arr.size * sizeof(Int32)).to_u64)
      end
      buf.raw.as(Int32*).copy_from(arr.to_unsafe, arr.size)
      unless Num::ClContext.instance.fine_grain_svm_supported?
        Cl.unmap_svm(Num::ClContext.instance.queue, buf.raw)
      end
    else
      Cl.write(Num::ClContext.instance.queue, arr.to_unsafe, buf, (arr.size * sizeof(Int32)).to_u64)
    end
    buf
  end

  private def free_buffer(buf : LibCL::ClMem | Cl::SVMPointer)
    if buf.is_a?(Cl::SVMPointer)
      LibCL.cl_svm_free(Num::ClContext.instance.context, buf.raw)
    else
      Cl.release_buffer(buf)
    end
  end

  def finalize
    free_buffer(@data)
    free_buffer(@shape)
    free_buffer(@strides)
  end
end

class Tensor(T, S)
  def self.from_storage(data : S, shape : Array(Int32), strides : Array(Int32), offset : Int32) : Tensor(T, S)
    new(data, shape, strides, offset, T)
  end

  # Returns a new Tensor pointing to a sub-buffer (zero-copy) of the current Tensor.
  # This requires the device memory offset (offset in bytes) to be aligned to the device's
  # CL_DEVICE_MEM_BASE_ADDR_ALIGN.
  def sub_tensor(origin_elements : Int, shape : Array(Int), strides : Array(Int)) : Tensor(T, S)
    {% if S == OCL(T) %}
      data_store = @data
      if data_store.is_a?(OCL(T))
        data_mem = data_store.data
        if data_mem.is_a?(LibCL::ClMem)
          byte_offset = (origin_elements * sizeof(T)).to_u64

          # Validation for sub-buffer alignment
          align = Cl.mem_base_addr_align(Num::ClContext.instance.device) / 8
          if byte_offset % align != 0
            raise "Sub-tensor byte offset (#{byte_offset}) must be aligned to #{align} bytes for this device"
          end

          byte_size = (shape.product * sizeof(T)).to_u64
          sub_mem = Cl.create_sub_buffer(data_mem, byte_offset, byte_size)
          new_storage = OCL(T).new(sub_mem, shape, strides)
          Tensor.new(new_storage, shape.map(&.to_i32), strides.map(&.to_i32), 0, T)
        else
          raise "Sub-tensor is only supported for OpenCL non-SVM backend"
        end
      else
        raise "Sub-tensor is only supported for OpenCL backend"
      end
    {% else %}
      raise "Sub-tensor is only supported on OpenCL backend"
    {% end %}
  end
end
