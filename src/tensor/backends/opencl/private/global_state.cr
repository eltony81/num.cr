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

require "opencl"

# :nodoc:
class Num::Internal::ClInfo
  getter device : LibCL::ClDeviceId
  getter context : LibCL::ClContext
  getter queue : LibCL::ClCommandQueue
  property? svm_supported : Bool
  property? fine_grain_svm_supported : Bool

  def initialize(@device : LibCL::ClDeviceId, @context : LibCL::ClContext, @queue : LibCL::ClCommandQueue)
    @svm_supported = Cl.svm_supported?(@device)
    @fine_grain_svm_supported = Cl.supports_fine_grain_svm?(@device)
  end
end

# :nodoc:
class Num::ClContext
  def self.create_optimized_queue(context : LibCL::ClContext, device : LibCL::ClDeviceId) : LibCL::ClCommandQueue
    # Try out-of-order execution + high priority
    properties = [
      LibCL::CL_QUEUE_PROPERTIES,
      LibCL::CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE,
      LibCL::CL_QUEUE_PRIORITY_KHR,
      LibCL::CL_QUEUE_PRIORITY_HIGH_KHR,
      0_u64
    ]
    status = 0
    queue = LibCL.cl_create_command_queue_with_properties(context, device, properties, pointerof(status))
    if status == 0
      return queue
    end

    # Try out-of-order execution only
    properties = [
      LibCL::CL_QUEUE_PROPERTIES,
      LibCL::CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE,
      0_u64
    ]
    queue = LibCL.cl_create_command_queue_with_properties(context, device, properties, pointerof(status))
    if status == 0
      return queue
    end

    # Fallback to standard command queue
    Cl.command_queue_for(context, device)
  end

  class_getter instance : Num::Internal::ClInfo do
    platform = Cl.first_platform
    device = {% if flag?(:opencl_any) %}
               Cl.get_devices(platform)[0]
             {% else %}
               Cl.get_devices(platform, LibCL::CL_DEVICE_TYPE_GPU)[0]
             {% end %}
    context = Cl.create_context([device])
    queue = create_optimized_queue(context, device)
    Num::Internal::ClInfo.new(device, context, queue)
  end

  def self.set_device(device)
    context = Cl.create_context([device])
    queue = create_optimized_queue(context, device)
    @@instance = Num::Internal::ClInfo.new(device, context, queue)
  end
end
