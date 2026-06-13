require "../src/num"
require "benchmark"

# Matrix size: 128 x 128
N = 128

# ------------------------------------------------------------------
# Method 1: Naive triple-nested loop matrix multiplication
# Complexity: O(n^3) with no hardware optimizations
# ------------------------------------------------------------------
def naive_matmul(a : Array(Array(Float64)), b : Array(Array(Float64)), n : Int32) : Array(Array(Float64))
  c = Array.new(n) { Array.new(n, 0.0) }
  n.times do |i|
    n.times do |j|
      n.times do |k|
        c[i][j] += a[i][k] * b[k][j]
      end
    end
  end
  c
end

# Build flat Ruby-style 2D arrays for the naive approach
puts "=== Matrix Multiplication Benchmark (#{N}x#{N}) ==="
puts ""

naive_a = Array.new(N) { Array.new(N) { rand } }
naive_b = Array.new(N) { Array.new(N) { rand } }

# Build num.cr Tensors for the vectorized approach
tensor_a = Tensor(Float64, CPU(Float64)).new([N, N]) { rand }
tensor_b = Tensor(Float64, CPU(Float64)).new([N, N]) { rand }

Benchmark.ips do |x|
  x.report("Naive nested loops (Array)") do
    naive_matmul(naive_a, naive_b, N)
  end

  x.report("Tensor matmul (BLAS)") do
    tensor_a.matmul(tensor_b)
  end
end

puts ""
puts "=== Correctness Check: First element of each result ==="

# Run each once
naive_result = naive_matmul(naive_a, naive_b, N)
tensor_result = tensor_a.matmul(tensor_b)

puts "Naive   C[0,0] = #{naive_result[0][0].round(6)}"
puts "Tensor  C[0,0] = #{tensor_result.to_unsafe[0].round(6)}"
puts ""
puts "=== Why is BLAS faster? ==="
puts "  1. SIMD: Processes 4-8 Float64 values per CPU clock cycle"
puts "  2. Cache Blocking: Tiles matrices into L1/L2 cache-sized chunks"
puts "  3. Loop Unrolling: Compiler unrolls inner loops to reduce branch overhead"
puts "  4. Zero GC Pressure: No heap allocation for intermediate arrays"
