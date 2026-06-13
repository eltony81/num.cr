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

# 2. Create a Table and write to a Parquet file
puts "--- 2. Creating Table and Writing to Parquet ---"
schema = Arrow::Schema.new([
  Arrow::Field.new("tensor_values", Arrow::DataType.int32)
])
arrow_arr = Arrow::Int32Array.new([100, 200, 300, 400, 500])
table = Arrow::Table.new(schema, [arrow_arr])

# Write table to Parquet file
filename = "./example_output.parquet"
puts "Writing Arrow Table to #{filename}..."
begin
  writer = Arrow::ParquetWriter.new(schema, filename)
  writer.write(table)
  writer.close
  puts "Parquet file successfully written!"
  File.delete(filename) if File.exists?(filename)
rescue ex
  puts "Parquet writing failed (is parquet-glib missing on your system?): #{ex.message}"
end
puts ""

# 3. Export Array using the C Data ABI Interface
puts "--- 3. C Data Interface ABI Export ---"
c_arr = Pointer(Void).null
c_schema = Pointer(Void).null

begin
  # Export the array structure to C ABI pointer representations
  arrow_arr.export(pointerof(c_arr), pointerof(c_schema))
  puts "Array exported successfully to C ABI structures!"
  
  # Import it back via the ABI
  imported_arr = Arrow::Array.import(c_arr, Arrow::DataType.int32)
  puts "Imported array length via C ABI: #{imported_arr.length}"
rescue ex
  puts "C Data ABI operation failed: #{ex.message}"
end
puts ""

# 4. Initialize Compute Engine
puts "--- 4. Compute Engine ---"
begin
  Arrow.initialize_compute
  puts "Arrow Compute Engine successfully initialized!"
rescue ex
  puts "Compute engine initialization failed: #{ex.message}"
end
