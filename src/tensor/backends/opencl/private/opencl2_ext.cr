# OpenCL 2.0+ Extensions for num.cr
#
# The upstream opencl.cr shard (v0.4.0) does not expose OpenCL 2.0 APIs.
# This file extends LibCL and Cl with the missing bindings needed by the
# SVM, out-of-order queue, and sub-buffer features introduced from v1.8.0.

{% if flag?(:darwin) %}
  @[Link(framework: "OpenCL")]
{% else %}
  @[Link("OpenCL")]
{% end %}
lib LibCL
  # -----------------------------------------------------------------------
  # OpenCL 2.0 Device Query
  # -----------------------------------------------------------------------
  CL_DEVICE_SVM_CAPABILITIES = 0x1053_u32

  # -----------------------------------------------------------------------
  # Command Queue Properties (OpenCL 2.0)
  # Used by clCreateCommandQueueWithProperties to pass a properties array.
  # -----------------------------------------------------------------------
  CL_QUEUE_PROPERTIES                   = 0x9013_u64
  CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE = 1_u64
  CL_QUEUE_PRIORITY_KHR                 = 0x1044_u64
  CL_QUEUE_PRIORITY_HIGH_KHR            = 1_u64

  # Creates a command queue using the OpenCL 2.0 properties list API.
  # Falls back gracefully if the driver does not support it.
  fun cl_create_command_queue_with_properties = clCreateCommandQueueWithProperties(
    context     : ClContext,
    device      : ClDeviceId,
    properties  : UInt64*,
    errcode_ret : ClInt*
  ) : ClCommandQueue

  # -----------------------------------------------------------------------
  # SVM (Shared Virtual Memory) – OpenCL 2.0
  # -----------------------------------------------------------------------
  fun cl_svm_alloc = clSVMAlloc(
    context   : ClContext,
    flags     : UInt64,
    size      : LibC::SizeT,
    alignment : ClUint
  ) : Void*

  fun cl_svm_free = clSVMFree(context : ClContext, svm_pointer : Void*) : Void

  fun cl_enqueue_svm_map = clEnqueueSVMMap(
    command_queue           : ClCommandQueue,
    blocking_map            : ClInt,
    flags                   : UInt64,
    svm_ptr                 : Void*,
    size                    : LibC::SizeT,
    num_events_in_wait_list : ClUint,
    event_wait_list         : ClEvent*,
    event                   : ClEvent*
  ) : ClInt

  fun cl_enqueue_svm_unmap = clEnqueueSVMUnmap(
    command_queue           : ClCommandQueue,
    svm_ptr                 : Void*,
    num_events_in_wait_list : ClUint,
    event_wait_list         : ClEvent*,
    event                   : ClEvent*
  ) : ClInt

  # -----------------------------------------------------------------------
  # Map Flags (used by SVM map operations)
  # -----------------------------------------------------------------------
  enum ClMapFlags : UInt64
    READ       = 1
    WRITE      = 2
    WRITE_INVALIDATE_REGION = 4
  end

  # -----------------------------------------------------------------------
  # Sub-buffer (OpenCL 1.1+)
  # -----------------------------------------------------------------------
  CL_BUFFER_CREATE_TYPE_REGION = 0x1220_i32

  struct ClBufferRegion
    origin : LibC::SizeT
    size   : LibC::SizeT
  end

  fun cl_create_sub_buffer = clCreateSubBuffer(
    buffer             : ClMem,
    flags              : ClMemFlags,
    buffer_create_type : ClInt,
    buffer_create_info : Void*,
    errcode_ret        : ClInt*
  ) : ClMem
end

# Extend the Cl convenience module with OpenCL 2.0 helper methods.
module Cl
  # Wraps a raw SVM (Shared Virtual Memory) pointer returned by clSVMAlloc.
  # It behaves as an opaque handle carrying the Void* pointer that can be
  # mapped / unmapped by the host and accessed directly by the device.
  class SVMPointer
    getter raw : Void*

    def initialize(@raw : Void*)
    end
  end

  # Returns true if the given OpenCL device supports at least Coarse-Grain
  # Buffer SVM (capabilities bit mask > 0).
  def svm_supported?(device : LibCL::ClDeviceId) : Bool
    caps = 0_u64
    status = LibCL.cl_get_device_info(
      device,
      LibCL::ClDeviceInfo.new(LibCL::CL_DEVICE_SVM_CAPABILITIES),
      sizeof(UInt64),
      pointerof(caps).as(Void*),
      nil
    )
    status == 0 && caps > 0
  rescue
    false
  end

  # Maps an SVM region into host-accessible memory.
  #
  # * *queue*    – the OpenCL command queue
  # * *blocking* – `LibCL::CL_TRUE` to block until mapping is done
  # * *flags*    – `LibCL::ClMapFlags` (READ, WRITE, …)
  # * *ptr*      – the SVM Void* pointer to map
  # * *size*     – number of bytes to map
  def map_svm(
    queue    : LibCL::ClCommandQueue,
    blocking : Int32,
    flags    : UInt64,
    ptr      : Void*,
    size     : UInt64
  )
    rc = LibCL.cl_enqueue_svm_map(queue, blocking, flags, ptr, size, 0_u32, nil, nil)
    Cl.check(rc)
  end

  # Unmaps a previously mapped SVM region, making it device-accessible again.
  def unmap_svm(queue : LibCL::ClCommandQueue, ptr : Void*)
    rc = LibCL.cl_enqueue_svm_unmap(queue, ptr, 0_u32, nil, nil)
    Cl.check(rc)
  end

  # Creates an OpenCL sub-buffer that aliases a byte range of *buffer*.
  # The *byte_offset* must be aligned to the device's `CL_DEVICE_MEM_BASE_ADDR_ALIGN`.
  def create_sub_buffer(
    buffer      : LibCL::ClMem,
    byte_offset : UInt64,
    byte_size   : UInt64
  ) : LibCL::ClMem
    region = LibCL::ClBufferRegion.new(origin: byte_offset, size: byte_size)
    status = 0
    sub_buf = LibCL.cl_create_sub_buffer(
      buffer,
      LibCL::ClMemFlags::READ_WRITE,
      LibCL::CL_BUFFER_CREATE_TYPE_REGION,
      pointerof(region).as(Void*),
      pointerof(status)
    )
    Cl.check(status)
    sub_buf
  end
end
