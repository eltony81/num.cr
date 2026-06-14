require "num"

# Example: Using advanced OpenCL features provided by eltony81/opencl.cr fork
#
# This example demonstrates:
# 1. Advanced Diagnostics (Num.opencl_info)
# 2. Performance benefits of Fine-Grained SVM (Automatic in v1.29+)
# 3. Sub-buffer alignment validation

puts "--- OpenCL Diagnostics ---"
puts Num.opencl_info

# Create a Tensor on OpenCL
a = Tensor.new([1024, 1024], device: OCL) { |i| i.to_f32 }

# Perform an operation
b = a * 2.0

puts "\nTensor created and multiplied on GPU."
puts "Backend: #{b.data.class}"

# Demonstrate sub-tensor creation (Zero-Copy)
# Note: The fork automatically validates alignment using Cl.mem_base_addr_align
begin
  # Attempt to create a sub-tensor
  # Offset must be aligned to the device's base address alignment
  sub = a.sub_tensor(0, [512, 512], a.strides)
  puts "\nSub-tensor created successfully (Zero-Copy)."
rescue e
  puts "\nSub-tensor creation failed: #{e.message}"
  puts "This is expected if the offset is not aligned to the device's requirements."
end

# The use of Fine-Grained SVM (if supported) is automatic.
# In OCL(T) initialization and data access, num.cr now detects
# fine_grain_svm_supported? and skips map/unmap operations,
# significantly reducing overhead for small, frequent transfers.
if Num::ClContext.instance.fine_grain_svm_supported?
  puts "\nOptimization Active: Using Fine-Grained SVM to bypass map/unmap overhead."
else
  puts "\nNote: Device does not support Fine-Grained SVM, falling back to Coarse-Grained SVM or Standard Buffers."
end
