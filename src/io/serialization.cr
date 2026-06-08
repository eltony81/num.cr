# Binary Serialization & Text I/O Extensions

class Tensor(T, S)
  # Save tensor metadata and buffer elements in a platform-independent flat binary format
  def save(filename : String) : Nil
    File.open(filename, "wb") do |file|
      r = self.rank.to_i32
      file.write(Bytes.new(pointerof(r).as(UInt8*), 4))
      
      self.shape.each do |s_val|
        s_i32 = s_val.to_i32
        file.write(Bytes.new(pointerof(s_i32).as(UInt8*), 4))
      end
      
      sz = self.size.to_i32
      file.write(Bytes.new(pointerof(sz).as(UInt8*), 4))
      
      # Ensure data is contiguous before dumping raw buffer
      contiguous_tensor = self.dup
      bytes_size = contiguous_tensor.size * sizeof(T)
      file.write(Bytes.new(contiguous_tensor.to_unsafe.as(UInt8*), bytes_size))
    end
  end

  # Load a tensor from our custom flat binary format
  def self.load(filename : String) : Tensor(T, S)
    File.open(filename, "rb") do |file|
      var_rank = 0_i32
      file.read(Bytes.new(pointerof(var_rank).as(UInt8*), 4))
      
      shape = Array(Int32).new(var_rank)
      var_rank.times do
        s_val = 0_i32
        file.read(Bytes.new(pointerof(s_val).as(UInt8*), 4))
        shape << s_val
      end
      
      sz = 0_i32
      file.read(Bytes.new(pointerof(sz).as(UInt8*), 4))
      
      result = Tensor(T, S).zeros(shape)
      bytes_size = sz * sizeof(T)
      file.read(Bytes.new(result.to_unsafe.as(UInt8*), bytes_size))
      result
    end
  end

  # Load standard text matrices (e.g., CSV, TSV) from a file as a Float64 CPU Tensor
  def self.loadtxt(filename : String, separator = ',') : Tensor(Float64, CPU(Float64))
    lines = File.read_lines(filename)
    parsed_rows = [] of Array(Float64)
    
    lines.each do |line|
      line_trimmed = line.strip
      next if line_trimmed.empty? || line_trimmed.starts_with?('#')
      parts = line_trimmed.split(separator)
      row_vals = parts.map &.strip.to_f64
      parsed_rows << row_vals
    end

    if parsed_rows.empty?
      raise Num::Exceptions::ValueError.new("loadtxt: No data found in file")
    end

    num_rows = parsed_rows.size
    num_cols = parsed_rows[0].size
    parsed_rows.each do |r|
      if r.size != num_cols
        raise Num::Exceptions::ValueError.new("loadtxt: Rows must have equal number of columns")
      end
    end

    result = Tensor(Float64, CPU(Float64)).zeros([num_rows, num_cols])
    num_rows.times do |i|
      num_cols.times do |j|
        result.to_unsafe[i * num_cols + j] = parsed_rows[i][j]
      end
    end
    result
  end
end
