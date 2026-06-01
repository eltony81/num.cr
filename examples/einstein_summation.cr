# This example demonstrates using the Einstein Summation (`einsum`)
# conventions in num.cr to perform various tensor contractions and operations
# using subscript strings.
#
# To run this example:
#   crystal run examples/einstein_summation.cr

require "../src/num"

# =====================================================================
# 1. Basic Vector Operations
# =====================================================================
puts "=== 1. Vector Operations via einsum ==="

v1 = [1.0, 2.0, 3.0].to_tensor
v2 = [4.0, 5.0, 6.0].to_tensor

# Vector Inner Product (dot product): sum_i v1_i * v2_i
# Subscript: "i,i" -> sums over the matching 'i' index
inner = Num::Einsum.einsum("i,i", v1, v2)
puts "Vector 1: #{v1}"
puts "Vector 2: #{v2}"
puts "Vector Inner Product (i,i): #{inner.value}\n\n"

# Vector Outer Product: out_ij = v1_i * v2_j
# Subscript: "i,j->ij"
outer = Num::Einsum.einsum("i,j->ij", v1, v2)
puts "Vector Outer Product (i,j->ij):"
puts outer
puts "\n"


# =====================================================================
# 2. Matrix Trace & Diagonal Extraction
# =====================================================================
puts "=== 2. Matrix Trace & Diagonal ==="

matrix = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]].to_tensor
puts "Matrix A:"
puts matrix

# Extract Diagonal: diag_i = A_ii
# Subscript: "ii->i"
diagonal = Num::Einsum.einsum("ii->i", matrix)
puts "\nDiagonal extraction (ii->i):"
puts diagonal

# Matrix Trace (sum of diagonal): trace = sum_i A_ii
# Subscript: "ii" (omitting the arrow sums over the repeated subscript)
trace = Num::Einsum.einsum("ii", matrix)
puts "\nMatrix Trace (ii): #{trace.value}"
puts "\n"


# =====================================================================
# 3. Matrix-Vector & Matrix-Matrix Multiplication
# =====================================================================
puts "=== 3. Matrix Multiplication via einsum ==="

a = [[1.0, 2.0], [3.0, 4.0]].to_tensor
b = [[5.0, 6.0], [7.0, 8.0]].to_tensor
v = [10.0, 20.0].to_tensor

puts "Matrix A:\n#{a}"
puts "Matrix B:\n#{b}"
puts "Vector v: #{v}"

# Matrix-Vector Multiplication: out_i = sum_j A_ij * v_j
# Subscript: "ij,j->i"
mv_prod = Num::Einsum.einsum("ij,j->i", a, v)
puts "\nMatrix-Vector Multiplication (ij,j->i):"
puts mv_prod

# Matrix-Matrix Multiplication: out_ik = sum_j A_ij * B_jk
# Subscript: "ij,jk->ik"
mm_prod = Num::Einsum.einsum("ij,jk->ik", a, b)
puts "\nMatrix-Matrix Multiplication (ij,jk->ik):"
puts mm_prod

# Element-wise (Hadamard) multiplication: out_ij = A_ij * B_ij
# Subscript: "ij,ij->ij"
hadamard = Num::Einsum.einsum("ij,ij->ij", a, b)
puts "\nElement-wise multiplication (ij,ij->ij):"
puts hadamard
puts "\n"


# =====================================================================
# 4. Multi-dimensional Tensor Contractions (Batch Operations)
# =====================================================================
puts "=== 4. Batch Matrix Multiplication (Tensor Contraction) ==="

# We have a batch of 2 matrices, each of size 2x3, and another batch of 2 matrices of size 3x2.
# Batch MM: out_bij = sum_k A_bik * B_bkj
# Subscript: "bik,bkj->bij"
batch_a = Tensor(Float64, CPU(Float64)).new([2, 2, 3]) { |i| i.to_f + 1.0 }
batch_b = Tensor(Float64, CPU(Float64)).new([2, 3, 2]) { |i| i.to_f + 1.0 }

batch_prod = Num::Einsum.einsum("bik,bkj->bij", batch_a, batch_b)
puts "Batch A shape: #{batch_a.shape}"
puts "Batch B shape: #{batch_b.shape}"
puts "Batch product shape (bik,bkj->bij): #{batch_prod.shape}"
puts "\nBatch Product:"
puts batch_prod
