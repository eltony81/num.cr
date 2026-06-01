# This example demonstrates the advanced linear algebra features
# introduced in the 1.7.0 release of num.cr.
#
# To run this example:
#   crystal run examples/advanced_linear_algebra.cr
# Or with multithreading enabled:
#   crystal run -Dpreview_mt examples/advanced_linear_algebra.cr

require "../src/num"

# =====================================================================
# 1. Schur Decomposition
# =====================================================================
puts "=== 1. Schur Decomposition ==="
# Factorizes a square matrix A into Z * T * Z^T
# where T is quasi-upper triangular (Schur form) and Z is orthogonal.
a = [[1.0, 2.0], [3.0, 4.0]].to_tensor
t, z = a.schur

puts "Original Matrix A:"
puts a
puts "\nQuasi-triangular Schur form T:"
puts t
puts "\nOrthogonal Matrix Z (Schur vectors):"
puts z

# Verify Z * T * Z^T approx = A
reconstructed = z.matmul(t).matmul(z.transpose)
puts "\nReconstructed A (Z * T * Z^T):"
puts reconstructed
puts "\n"


# =====================================================================
# 2. Sylvester & Lyapunov Equation Solvers
# =====================================================================
puts "=== 2. Sylvester Equation Solver ==="
# Solves: A * X + X * B = C
# Useful in state feedback control design, pole placement, etc.
syl_a = [[1.0, 2.0], [3.0, 4.0]].to_tensor
syl_b = [[5.0, 6.0], [7.0, 8.0]].to_tensor
syl_c = [[9.0, 10.0], [11.0, 12.0]].to_tensor

x_syl = Tensor.sylvester(syl_a, syl_b, syl_c)
puts "Solution X to A*X + X*B = C:"
puts x_syl

# Verify A * X + X * B = C
syl_verify = syl_a.matmul(x_syl) + x_syl.matmul(syl_b)
puts "\nVerification (A*X + X*B):"
puts syl_verify
puts "\n"

puts "=== 3. Continuous Lyapunov Equation Solver ==="
# Solves: A * X + X * A^T = Q
# Crucial for stability analysis, checking controllability/observability Gramians.
lyap_a = [[-3.0, 1.0], [0.0, -2.0]].to_tensor
lyap_q = [[1.0, 0.0], [0.0, 1.0]].to_tensor

x_lyap = Tensor.lyapunov(lyap_a, lyap_q)
puts "Solution X to A*X + X*A^T = Q:"
puts x_lyap

# Verify A * X + X * A^T = Q
lyap_verify = lyap_a.matmul(x_lyap) + x_lyap.matmul(lyap_a.transpose)
puts "\nVerification (A*X + X*A^T):"
puts lyap_verify
puts "\n"


# =====================================================================
# 3. Higham Padé Matrix Exponential (expm)
# =====================================================================
puts "=== 4. Higham Padé Matrix Exponential (expm) ==="
# Computes the matrix exponential using scaling-and-squaring with Padé approximants.
# Essential for continuous-to-discrete state-space conversion.
sys_matrix = [[0.0, 1.0], [-2.0, -3.0]].to_tensor
exp_sys = sys_matrix.expm

puts "Matrix Exponential (expm) of [[0, 1], [-2, -3]]:"
puts exp_sys
puts "\n"


# =====================================================================
# 4. Offset Diagonal Views
# =====================================================================
puts "=== 5. Offset Diagonal Views ==="
# zero-copy view of offset sub-diagonals (both upper and lower).
matrix_3x3 = [[1, 2, 3], [4, 5, 6], [7, 8, 9]].to_tensor

puts "Matrix:"
puts matrix_3x3

# Get diagonal with offset = 1 (upper sub-diagonal)
puts "\nDiagonal with offset = 1:"
puts matrix_3x3.diagonal(1)

# Get diagonal with offset = -1 (lower sub-diagonal)
puts "\nDiagonal with offset = -1:"
puts matrix_3x3.diagonal(-1)
puts "\n"


# =====================================================================
# 5. Parallel Mapping / Multithreaded Iterations
# =====================================================================
puts "=== 6. Parallel Mapping Iterations ==="
# When compiled with `-Dpreview_mt` and size >= 100_000, 
# CPU mapping executes across multiple system threads in parallel.

large_size = 200_000
puts "Initializing a large contiguous tensor of size #{large_size}..."
large_tensor = Tensor(Float64, CPU(Float64)).new([large_size]) { |i| i.to_f }

# Warmup or measure time
start_time = Time.monotonic
result = large_tensor.map { |x| Math.sin(x) * Math.cos(x) }
elapsed = Time.monotonic - start_time

puts "Mapped elements: #{result.size}"
puts "Time elapsed: #{elapsed.total_milliseconds.round(2)} ms"
if ENV.has_key?("CRYSTAL_WORKERS") || {% if flag?(:preview_mt) %} true {% else %} false {% end %}
  puts "Running in multithreaded mode!"
else
  puts "Running in single-threaded mode (compile with -Dpreview_mt for parallelism)."
end
