# This example demonstrates the fundamental mathematical operations,
# random creation, slicing/views, and linear algebra decomposition utilities
# available in num.cr.
#
# To run this example:
#   crystal run examples/mathematical_utilities.cr

require "../src/num"

# =====================================================================
# 1. Tensor Creation & Initialization Utilities
# =====================================================================
puts "=== 1. Tensor Creation & Initialization ==="

# Create from arrays or nested arrays
t1 = [[1.0, 2.0], [3.0, 4.0]].to_tensor
puts "From nested array:"
puts t1
puts "Shape: #{t1.shape}, Storage: #{t1.class}\n\n"

# Initializing with zeros, ones, or identity
zeros = Tensor(Float64, CPU(Float64)).zeros([2, 3])
ones = Tensor(Float64, CPU(Float64)).ones([3, 2])
identity = Tensor(Float64, CPU(Float64)).eye(3)

puts "Zeros (2x3):"
puts zeros
puts "\nOnes (3x2):"
puts ones
puts "\nIdentity (3x3):"
puts identity
puts "\n"

# Random initialization
# - Uniform distribution on [0, 1)
uniform_rand = Tensor(Float64, CPU(Float64)).rand([2, 2])
# - Standard Normal distribution (mean=0, std=1)
normal_rand = Tensor(Float64, CPU(Float64)).normal([2, 2])

puts "Uniform Random [0, 1):"
puts uniform_rand
puts "\nStandard Normal Random:"
puts normal_rand
puts "\n"


# =====================================================================
# 2. Slicing, Reshaping, and Transposition
# =====================================================================
puts "=== 2. Slicing, Reshaping & Transposition ==="

# Arrange consecutive values
a = Tensor.new([3, 4]) { |i| i.to_f }
puts "Original 3x4 Tensor:"
puts a

# Reshape
reshaped = a.reshape([2, 6])
puts "\nReshaped to 2x6:"
puts reshaped

# Transpose
transposed = a.transpose
puts "\nTransposed (4x3):"
puts transposed

# Diagonal views with offset
diag_0 = a.diagonal(0)
diag_1 = a.diagonal(1)
diag_minus_1 = a.diagonal(-1)

puts "\nMain Diagonal (offset=0):"
puts diag_0
puts "\nUpper Diagonal (offset=1):"
puts diag_1
puts "\nLower Diagonal (offset=-1):"
puts diag_minus_1
puts "\n"


# =====================================================================
# 3. Element-wise Math Operations
# =====================================================================
puts "=== 3. Element-wise Math Operations ==="

x = [1.0, 2.0, 3.0].to_tensor
y = [4.0, 5.0, 6.0].to_tensor

puts "x: #{x}"
puts "y: #{y}"

# Arithmetic operators
puts "\nx + y:"
puts x + y
puts "\nx * y (Element-wise):"
puts x * y
puts "\nx ** 2 (Element-wise power):"
puts x ** 2

# Tracing mathematical functions
angles = [0.0, Math::PI / 2, Math::PI].to_tensor
puts "\nAngles (rad):"
puts angles
puts "\nSine (element-wise):"
puts angles.sin
puts "\nCosine (element-wise):"
puts angles.cos
puts "\nExponential of x:"
puts x.exp
puts "\nSquare root of y:"
puts y.sqrt
puts "\n"


# =====================================================================
# 4. Statistical Reductions
# =====================================================================
puts "=== 4. Statistical Reductions ==="

r = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]].to_tensor
puts "r matrix:"
puts r

# Global sum, mean, max
puts "\nGlobal sum: #{r.sum}"
puts "Global mean: #{r.mean}"
puts "Global maximum: #{r.max}"
puts "Global minimum: #{r.min}"

# Reductions along axes (requires passing the dimension)
# Note: Reductions in num.cr return a Tensor of reduced shape or scalar.
sum_axis_0 = r.sum(0)
sum_axis_1 = r.sum(1)

puts "\nSum along Axis 0 (column sums):"
puts sum_axis_0
puts "\nSum along Axis 1 (row sums):"
puts sum_axis_1
puts "\n"


# =====================================================================
# 5. Classic Linear Algebra Decompositions
# =====================================================================
puts "=== 5. Classic Linear Algebra Decompositions ==="

# Cholesky Decomposition (Symmetric Positive Definite Matrix)
spd = [[4.0, 12.0, -16.0], [12.0, 37.0, -43.0], [-16.0, -43.0, 98.0]].to_tensor
spd_fortran = spd.dup(Num::ColMajor)
spd_fortran.cholesky!(lower: true)
puts "Symmetric Positive Definite Matrix:"
puts spd
puts "\nCholesky factor L (lower: true):"
puts spd_fortran
puts "\nReconstructed (L * L^T):"
puts spd_fortran.matmul(spd_fortran.transpose)
puts "\n"

# QR Decomposition
q_matrix = [[12.0, -51.0, 4.0], [6.0, 167.0, -68.0], [-4.0, 24.0, -41.0]].to_tensor
q, r_qr = q_matrix.qr
puts "Matrix A for QR:"
puts q_matrix
puts "\nOrthogonal Q:"
puts q
puts "\nUpper Triangular R:"
puts r_qr
puts "\nReconstructed A (Q * R):"
puts q.matmul(r_qr)
puts "\n"

# Singular Value Decomposition (SVD)
svd_matrix = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]].to_tensor
u, s, vt = svd_matrix.svd
puts "Matrix A for SVD:"
puts svd_matrix
puts "\nU (Left singular vectors):"
puts u
puts "\nS (Singular values):"
puts s
puts "\nVT (Right singular vectors transposed):"
puts vt

# Mathematical Matrix Rank (using SVD)
# Note: In num.cr, `tensor.rank` returns the number of tensor dimensions (axes).
# To find the mathematical rank of a matrix, count the non-zero singular values.
tol = 1e-12
math_rank = 0
s.each { |val| math_rank += 1 if val.abs > tol }
puts "\nTensor dimensions (rank property in num.cr): #{svd_matrix.rank}"
puts "Mathematical Matrix Rank (from SVD): #{math_rank}"
puts "\n"

# Diagonalization (Eigenvalues & Eigenvectors)
sym_matrix = [[0.0, 1.0], [1.0, 1.0]].to_tensor
w, v = sym_matrix.eigh
puts "Symmetric Matrix A for Diagonalization:"
puts sym_matrix
puts "\nEigenvalues (w):"
puts w
puts "\nEigenvectors (v):"
puts v
puts "\n"

# Matrix Inverse, Determinant, and System Solver
a_sys = [[1.0, 2.0], [3.0, 4.0]].to_tensor
b_sys = [5.0, 11.0].to_tensor

puts "Coefficient matrix A:"
puts a_sys
puts "Determinant of A: #{a_sys.det}"
puts "Inverse of A:"
puts a_sys.inv

# Solve A * x = b
x_sys = a_sys.solve(b_sys)
puts "\nSolved solution x to A * x = b:"
puts x_sys
puts "\nVerification (A * x):"
puts a_sys.matmul(x_sys.reshape([2, 1])).reshape([2])
puts "\n"
