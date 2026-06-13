# This example demonstrates and benchmarks the new OpenCL 2.0+ capabilities
# integrated via opencl.cr v0.4.0 in num.cr:
#   1. Shared Virtual Memory (SVM) allocations for zero-copy host/device sharing.
#   2. Out-of-Order Command Queues for optimized parallel GPU kernel dispatch.
#
# Compile with:
#   crystal build -Dopencl --release examples/opencl_features_demo.cr
#
# Note: Requires a system with OpenCL 2.0+ compatible GPU and runtimes.

require "../src/api"
require "benchmark"

puts "=========================================================="
puts "  num.cr OpenCL 2.0+ Feature Showcase & SVM Benchmark     "
puts "=========================================================="

# Check if OpenCL is supported and available in the current compile context
{% if flag?(:opencl) %}
  device = Num::ClContext.instance.device
  context = Num::ClContext.instance.context
  queue = Num::ClContext.instance.queue

  device_name = Cl.device_name(device)
  opencl_version = Cl.version(device)
  svm_support = Num::ClContext.instance.svm_supported?

  puts "Device Name:      #{device_name}"
  puts "OpenCL Version:   #{opencl_version}"
  puts "SVM Supported:    #{svm_support ? "Yes (clSVMAlloc enabled)" : "No (Using clCreateBuffer)"}"

  # Size of Tensors to operate on (1,000,000 elements)
  n = 1_000_000
  puts "\nAllocating Tensors of size: #{n} (Float32)"

  # 1. Standard Buffer Tensor allocation (Traditional OpenCL 1.x)
  # Behind the scenes, OCL(Float32) will use standard clCreateBuffer if SVM is not supported,
  # but we can force or simulate a comparison if SVM is supported.
  
  if svm_support
    puts "\nSince Shared Virtual Memory (SVM) is supported, num.cr automatically"
    puts "allocates OCL storage using SVM (zero-copy memory mapping)."
    puts "Let's benchmark the allocation and element-wise addition on the GPU."

    a_storage = OCL(Float32).new([n], Num::RowMajor, 0_f32)
    Cl.map_svm(Num::ClContext.instance.queue, LibCL::CL_TRUE, LibCL::ClMapFlags::WRITE.value, a_storage.data.as(Cl::SVMPointer).raw, (n * sizeof(Float32)).to_u64)
    a_storage.data.as(Cl::SVMPointer).raw.as(Float32*).map!(n) { |i| i.to_f32 }
    Cl.unmap_svm(Num::ClContext.instance.queue, a_storage.data.as(Cl::SVMPointer).raw)
    a = Tensor(Float32, OCL(Float32)).new(a_storage, [n], Num::RowMajor, Float32)

    b = Tensor(Float32, OCL(Float32)).new([n], 100.0_f32, OCL(Float32))

    puts "Memory address of a.data.data (SVM): #{a.data.data.as(Cl::SVMPointer).raw}"

    # Perform arithmetic on SVM Tensors
    puts "Computing a + b on OpenCL device..."
    c = a + b

    # Convert to CPU to print some values
    c_cpu = c.to_a
    puts "Verification (first 5 elements): #{c_cpu[0..4]}"

    Benchmark.ips do |x|
      x.report("OpenCL SVM Allocation + Math") do
        t1 = Tensor(Float32, OCL(Float32)).new([n], 1.0_f32, OCL(Float32))
        t2 = Tensor(Float32, OCL(Float32)).new([n], 100.0_f32, OCL(Float32))
        res = t1 + t2
        res.to_a # force readback
      end
    end
  else
    puts "\nSVM is not supported by your OpenCL runtime/device."
    puts "num.cr fallback: allocated traditional device buffers."
    
    a = Tensor(Float32, OCL(Float32)).new([n], 1.0_f32, OCL(Float32))
    b = Tensor(Float32, OCL(Float32)).new([n], 100.0_f32, OCL(Float32))
    c = a + b
    puts "Verification (first 5 elements): #{c.to_a[0..4]}"
  end

  # Demonstrate setting custom Out-of-Order Execution Queues
  puts "\nDemonstrating Command Queue Properties customization:"
  # We construct a custom out-of-order queue with properties
  begin
    properties = [
      LibCL::CL_QUEUE_PROPERTIES,
      LibCL::CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE | LibCL::CL_QUEUE_PROFILING_ENABLE,
      0_u64
    ]
    status = 0
    custom_queue = LibCL.cl_create_command_queue_with_properties(context, device, properties, pointerof(status))
    if status == 0
      puts "Successfully initialized custom Out-of-Order Command Queue with profiling!"
    else
      puts "Failed to initialize custom Out-of-Order Command Queue (status code: #{status})"
    end
  rescue e
    puts "Failed to create custom command queue: #{e.message}"
  end

{% else %}
  puts "\nThis example requires compilation with the -Dopencl flag:"
  puts "  crystal run -Dopencl examples/opencl_features_demo.cr"
{% end %}
