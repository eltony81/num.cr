# Example: Using the new Apache Arrow features in num.cr
# Run this example using: crystal run -Darrow examples/arrow_features_example.cr

require "../src/api"

# 1. Build a Tensor on CPU and place it on Arrow backend
puts "--- 1. Creating and Exporting Tensor ---"
t = Tensor(Int32, ARROW(Int32)).new([3, 3], device: ARROW(Int32)) { |i| i * 5 }
puts "Num::Tensor (ARROW-backed):"
puts t

# Export to Arrow::Tensor (Zero-copy)
arrow_tensor = t.to_arrow_tensor
puts "Exported to Arrow::Tensor:"
puts "  Dimensions (ndim): #{arrow_tensor.ndim}"
puts "  Total size: #{arrow_tensor.size}"
puts "  Shape: #{arrow_tensor.shape}"
puts "  Byte Strides: #{arrow_tensor.strides}"

# Import back from Arrow::Tensor
imported_t = Tensor(Int32, ARROW(Int32)).from_arrow_tensor(arrow_tensor)
puts "Imported back to Num::Tensor:"
puts imported_t
puts ""

# 2. Vectorized Math via Arrow Compute Engine
puts "--- 2. Vectorized Element-wise Math ---"
# This call delegates to Arrow Compute C++ engine under the hood
res_t = t + t
puts "Result of (t + t) using Arrow Compute Engine:"
puts res_t
puts ""

# 3. Create a Table and write to a Parquet/Feather file
puts "--- 3. Parquet and Feather File I/O ---"
schema = Arrow::Schema.new([
  Arrow::Field.new("tensor_values", Arrow::DataType.int32)
])
arrow_arr = Arrow::Int32Array.new([100, 200, 300, 400, 500])
table = Arrow::Table.new(schema, [arrow_arr])

# Write table to Parquet file
pq_name = "./example_output.parquet"
begin
  writer = Arrow::ParquetWriter.new(schema, pq_name)
  writer.write(table)
  writer.close
  puts "Parquet file successfully written!"
  File.delete(pq_name) if File.exists?(pq_name)
rescue ex
  puts "Parquet writing skipped/failed: #{ex.message}"
end

# Write table to Feather file
ft_name = "./example_output.feather"
begin
  f_writer = Arrow::FeatherWriter.new(ft_name)
  f_writer.write(table)
  f_writer.close
  puts "Feather file successfully written!"
  File.delete(ft_name) if File.exists?(ft_name)
rescue ex
  puts "Feather writing skipped/failed: #{ex.message}"
end
puts ""

# 4. Export Array using the C Data ABI Interface
puts "--- 4. C Data Interface ABI Export ---"
c_arr = Pointer(Void).null
c_schema = Pointer(Void).null

begin
  arrow_arr.export(pointerof(c_arr), pointerof(c_schema))
  puts "Array exported successfully to C ABI structures!"
  imported_arr = Arrow::Array.import(c_arr, Arrow::DataType.int32)
  puts "Imported array length via C ABI: #{imported_arr.length}"
rescue ex
  puts "C Data ABI operation failed: #{ex.message}"
end
puts ""

# 5. Initialize GPU Memory Sharing (CUDA)
puts "--- 5. CUDA GPU memory sharing ---"
begin
  manager = Arrow::CudaDeviceManager.new
  puts "CUDA Device Manager initialized! Total CUDA devices: #{manager.devices_count}"
  if manager.devices_count > 0
    ctx = manager.get_context(0)
    puts "Context created. Total allocated GPU size: #{ctx.allocated_size} bytes"
  end
rescue ex
  puts "CUDA GPU initialization skipped/failed: #{ex.message}"
end
puts ""

# 6. Initialize Flight Server
puts "--- 6. Flight Client/Server RPC ---"
begin
  server = Arrow::FlightServer.new
  server.listen(8200)
  puts "Flight Server successfully started on port #{server.port}!"
  server.shutdown
  puts "Flight Server successfully shut down!"
rescue ex
  puts "Flight Server setup skipped/failed: #{ex.message}"
end
