![num.cr](https://raw.githubusercontent.com/eltony81/bottle/rename/static/numcr_logo.png)

[![Join the chat at https://gitter.im/eltony81/bottle](https://badges.gitter.im/eltony81/bottle.svg)](https://gitter.im/eltony81/bottle?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)
![Crystal CI](https://github.com/eltony81/num.cr/workflows/Crystal%20CI/badge.svg)
![Version](https://img.shields.io/badge/version-1.24.3-blue)

Num.cr is the core shard needed for scientific computing with Crystal

- **Website:** https://eltony81.github.io/num.cr
- **API Documentation:** https://eltony81.github.io/num.cr/
- **Source code:** https://github.com/eltony81/num.cr
- **Bug reports:** https://github.com/eltony81/num.cr/issues

It provides:

- An n-dimensional `Tensor` data structure
- Efficient `map`, `reduce` and `accumulate` routines
- GPU accelerated routines backed by `OpenCL`
- Linear algebra routines backed by `LAPACK` and `BLAS`

## Fork Improvements (eltony81/num.cr)

This fork is maintained to support control theory applications and robust computations in the `cryspace` project. Key enhancements and improvements over the original `crystal-data/num.cr` include:

- **Complex Eigenvalue Support (`eigvals_c` & `eig_c`)**: Added methods in linear algebra to compute eigenvalues/eigenvectors and return a Complex Tensor (`Tensor(Complex, CPU(Complex))`), which is critical for stability and control system analysis in `cryspace`.
- **Dynamic LAPACK Workspace Queries**: Replaced hardcoded LAPACK workspace sizes with dynamic workspace queries (`lwork = -1`) inside eigenvalue/eigenvector routines to optimize memory allocation and stability.
- **Fixed CBLAS Signatures & Correct Linking**: 
  - Corrected parameter signatures in `cblas.cr` for `cblas_dtbmv`, `cblas_dtbsv`, and `cblas_dsymm` (ensuring appropriate double/float precision mapping).
  - Cleaned up implicit OpenBLAS library linking in `cblas.cr` to prevent collision issues across different platform/distro environments.
- **Robust LAPACK Solver Integration**: Resolved alignment/solver issues in `solve` (calling LAPACK solver like `sgesv`/`dgesv`).
- **Improved CI Pipelines**: Added support for standard Debian-based environments and official test containers by explicitly resolving dependency libraries (`openblas`, `clblast`, `arrow`, `atlas`, `libcblas-dev`).
- **Apache Arrow Features & Performance (eltony81/arrow.cr 1.3.0)**:
  - **Vectorized Compute Delegation**: Automatically offloads contiguous tensor arithmetic math operations (`add`, `subtract`, `multiply`, `divide`), in-place operations (`add!`, `subtract!`, `multiply!`, `divide!`), and unary `negate` directly to Arrow's SIMD-optimized C++ compute engine on the `ARROW` backend.
  - **Feather & Parquet File I/O**: Exposes `Arrow::FeatherWriter` and `Arrow::ParquetWriter` for ultra-fast dataset saving and loading.
  - **CUDA GPU Memory Sharing**: Zero-copy GPU memory sharing via `Arrow::CudaDeviceManager` and `Arrow::CudaBuffer`.
  - **Flight Client/Server RPC**: High-throughput distributed data streaming via `Arrow::FlightClient` and `Arrow::FlightServer`.
  - **C Data Interface ABI**: Implemented zero-copy export/import support via `Arrow::Array.import` and `Arrow::Array#export`.
  - **Raw Pointer Performance**: Bypassed GLib overhead for element-wise and buffer iteration by caching and exposing raw memory pointers (`#raw_pointer`).

## Prerequisites

`Num.cr` aims to be a scientific computing library written in pure Crystal.
All standard operations and data structures are written in Crystal.  Certain
routines, primarily linear algebra routines, are instead provided by a
`BLAS` or `LAPACK` implementation.

Several implementations can be used, including `Cblas`, `Openblas`, and the
`Accelerate` framework on Darwin systems.  For GPU accelerated `BLAS` routines,
the `ClBlast` library is required.

`Num.cr` also supports `Tensor`s stored on a `GPU`.  This is currently limited
to `OpenCL`, and a valid `OpenCL` installation and device(s) are required.

## Installation

Add this to your applications `shard.yml`

```yaml
dependencies:
  num:
    github: eltony81/num.cr
    version: ~> 1.24.0
```

### Vectorized SIMD Mode (Apache Arrow)

To enable SIMD offloading for Tensors utilizing the `ARROW` backend, compile your application with the `-Darrow` flag:

```bash
crystal build -Darrow --release src/your_app.cr
```

When `-Darrow` is enabled:
- Standard math operations (`+`, `-`, `*`, `/`), in-place operations (`add!`, `subtract!`, `multiply!`, `divide!`), and unary negation (`-`) on ARROW-backed Tensors are automatically offloaded to the C++ Apache Arrow Compute Engine, utilizing optimized SIMD execution (AVX2, AVX-512, or ARM Neon depending on hardware).
- File I/O for Parquet/Feather and CUDA device sharing is fully operational.

### Dynamic Backend Dispatch (CPU, Arrow SIMD, OpenCL GPU)

Starting in `v1.24.3`, `num.cr` supports an automatic runtime dispatch mechanism for CPU element-wise arithmetic operations (`+`, `-`, `*`, `/`) and negation (`-`). When compiling with backend flags (`-Darrow`, `-Dopencl`), standard CPU-allocated tensors will dynamically route execution paths to the most appropriate backend based on size thresholds:

- **OpenCL GPU Acceleration** (`-Dopencl` flag): If a CPU tensor has **$\ge 1,000,000$ elements** (and its datatype is supported by OpenCL like `Int32`, `UInt32`, `Float32`, `Float64`), it is automatically copied to the OpenCL GPU device, executed, and the result is copied back to CPU storage.
- **Apache Arrow SIMD Acceleration** (`-Darrow` flag): For medium-scale datasets with **$1,000 \le \text{size} < 1,000,000$ elements** (and contiguous shape), the operation is automatically wrapped and executed on Apache Arrow's vectorized C++ compute engine.
- **Standard CPU Loop**: For small datasets ($< 1,000$ elements), the operation falls back to standard CPU loops to bypass device memory overhead or API wrapping latencies.

#### Activation & Selection Matrix:

To compile your code, select the appropriate flags depending on your target workloads:

*   **Arrow SIMD only**: `crystal build -Darrow --release src/your_app.cr`
*   **OpenCL GPU only**: `crystal build -Dopencl --release src/your_app.cr`
*   **Hybrid (Both backends)**: `crystal build -Darrow -Dopencl --release src/your_app.cr` (routes dynamically based on size thresholds).


Several third-party libraries are required to use certain features of `Num.cr`.
They are:

- BLAS
- LAPACK
- OpenCL
- ClBlast
- NNPACK

While not at all required, they provide additional functionality than is
provided by the basic library.

### Dependency Versions

The library has been tested and run successfully with the following dependency and system library versions:

| Dependency | Type | Version | Note |
|---|---|---|---|
| **Crystal** | Language | `>= 1.0.0` | Core language runtime |
| **OpenBLAS** | System Library | `0.3.33` | Highly optimized BLAS/LAPACK implementation |
| **BLAS / CBLAS** | System Library | `3.12.1` | Basic Linear Algebra Subprograms |
| **LAPACK** | System Library | `3.12.1` | Linear Algebra Package |
| **OpenCL** | System Library | `2.3.4` (via `ocl-icd`) | GPU acceleration support |
| **opencl.cr** | Shard | `0.2.1` | Crystal bindings for OpenCL |
| **alea** | Shard | `0.3.0` | Random number generation library |
| **arrow.cr** | Shard | `1.3.0` | Apache Arrow bindings |

## Just show me the code

The core data structure implemented by `Num.cr` is the `Tensor`, an N-dimensional
data structure.  A `Tensor` supports slicing, mutation, permutation, reduction,
and accumulation.  A `Tensor` can be a view of another `Tensor`, and can support
either C-style or Fortran-style storage.

### Creation

There are many ways to initialize a `Tensor`.  Most creation methods can
allocate a `Tensor` backed by either `CPU` or `GPU` based storage.

```crystal
[1, 2, 3].to_tensor
Tensor.from_array [1, 2, 3]
Tensor(UInt8, CPU(UInt8)).zeros([3, 3, 2])
Tensor.random(0.0...1.0, [2, 2, 2])

Tensor(Float32, OCL(Float32)).zeros([3, 2, 2])
Tensor(Float64, OCL(Float64)).full([3, 4, 5], 3.8)
```

### Operations

A `Tensor` supports a wide variety of numerical operations.  Many of these
operations are provided by `Num.cr`, but any operation can be mapped across
one or more `Tensor`s using sophisticated broadcasted mapping routines.

```crystal
a = [1, 2, 3, 4].to_tensor
b = [[3, 4, 5, 6], [5, 6, 7, 8]].to_tensor

puts a + b

# a is broadcast to b's shape
# [[ 4,  6,  8, 10],
#  [ 6,  8, 10, 12]]
```

When operating on more than two `Tensor`s, it is recommended to use `map`
rather than builtin functions to avoid the allocation of intermediate
results.  All `map` operations support broadcasting.

```crystal
a = [1, 2, 3, 4].to_tensor
b = [[3, 4, 5, 6], [5, 6, 7, 8]].to_tensor
c = [3, 5, 7, 9].to_tensor

a.map(b, c) do |i, j, k|
  i + 2 / j + k * 3.5
end

# [[12.1667, 20     , 27.9   , 35.8333],
#  [11.9   , 19.8333, 27.7857, 35.75  ]]
```

### Mutation

`Tensor`s support flexible slicing and mutation operations.  Many of these
operations return views, not copies, so any changes made to the results might
also be reflected in the parent.

```crystal
a = Tensor.new([3, 2, 2]) { |i| i }

puts a.transpose

# [[[ 0,  4,  8],
#   [ 2,  6, 10]],
#
#  [[ 1,  5,  9],
#   [ 3,  7, 11]]]

puts a.reshape(6, 2)

# [[ 0,  1],
#  [ 2,  3],
#  [ 4,  5],
#  [ 6,  7],
#  [ 8,  9],
#  [10, 11]]

puts a[..., 1]

# [[ 2,  3],
#  [ 6,  7],
#  [10, 11]]

puts a[1..., {..., -1}]

# [[[ 6,  7],
#   [ 4,  5]],
#
#  [[10, 11],
#   [ 8,  9]]]

puts a[0, 1, 1].value

# 3
```

### Linear Algebra

`Tensor`s provide easy access to power Linear Algebra routines backed by
LAPACK and BLAS implementations, and ClBlast for GPU backed `Tensor`s.

```crystal
a = [[1, 2], [3, 4]].to_tensor.map &.to_f32

puts a.inv

# [[-2  , 1   ],
#  [1.5 , -0.5]]

puts a.eigvals

# [-0.372281, 5.37228  ]

puts a.matmul(a)

# [[7 , 10],
#  [15, 22]]

# --- Fork-Specific Features ---

# 1. Compute complex eigenvalues & eigenvectors (essential for stability/control theory)
b = [[0, -1], [1, 0]].to_tensor.map &.to_f64
puts b.eigvals_c
# [(0.0 + 1.0i), (0.0 - 1.0i)]

w, v = b.eig_c
puts w
# [(0.0 + 1.0i), (0.0 - 1.0i)]

# 2. Matrix Power (positive, negative, and zero exponents)
a = [[1.0, 2.0], [3.0, 4.0]].to_tensor
puts a.matrix_power(2)
# [[ 7, 10],
#  [15, 22]]

# 3. Moore-Penrose Pseudoinverse (pinv)
# Useful for finding least-squares solutions in MIMO systems
tall_matrix = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]].to_tensor
puts tall_matrix.pinv

# 4. Kronecker Product (kron)
# Used for solving Lyapunov & Sylvester equations
c = [[1.0, 2.0], [3.0, 4.0]].to_tensor
d = [[0.0, 5.0], [2.0, 1.0]].to_tensor
puts c.kron(d)

# 5. Matrix Exponential (expm)
# Upgraded to Higham Padé approximation (essential for continuous-to-discrete conversion)
sys_matrix = [[0.0, 1.0], [-2.0, -3.0]].to_tensor
puts sys_matrix.expm

# 6. Schur Decomposition (schur)
# Decomposes matrix into quasi-triangular and orthogonal matrices: A = Z * T * Z^T
a = [[1.0, 2.0], [3.0, 4.0]].to_tensor
t, z = a.schur

# 7. Sylvester & Lyapunov Solvers
# Solves Sylvester: A * X + X * B = C  and Lyapunov: A * X + X * A^T = Q
a = [[1.0, 2.0], [3.0, 4.0]].to_tensor
b = [[5.0, 6.0], [7.0, 8.0]].to_tensor
c = [[9.0, 10.0], [11.0, 12.0]].to_tensor
x = Tensor.sylvester(a, b, c)

# 8. Offset Diagonals
# Zero-copy views of offset sub-diagonals
a = [[1, 2, 3], [4, 5, 6], [7, 8, 9]].to_tensor
puts a.diagonal(1)  # => [2, 6]
puts a.diagonal(-1) # => [4, 8]
```

### Einstein Notation

For representing certain complex contractions of `Tensor`s, Einstein notation
can be used to simplify the operation.  For example, the following matrix
multiplication + summation operation:

```crystal
a = Tensor.new([30, 40, 50]) { |i| i * 1_f32 }
b = Tensor.new([40, 30, 20]) { |i| i * 1_f32 }

result = Float32Tensor.zeros([50, 20])
ny, nx = result.shape
b2 = b.swap_axes(0, 1)
ny.times do |k|
  nx.times do |l|
    result[k, l] = (a[..., ..., k] * b2[..., ..., l]).sum
  end
end
```

Can instead be represented in Einstein notiation as the following:

```crystal
Num::Einsum.einsum("ijk,jil->kl", a, b)
```

This can lead to performance improvements due to optimized contractions
on `Tensor`s.

```
einsum   2.22k   (450.41µs) (± 0.86%)   350kB/op        fastest
manual   117.52  (  8.51ms) (± 0.98%)  5.66MB/op  18.89× slower
```

### Machine Learning

`Num::Grad` provides a pure-crystal approach to find derivatives of
mathematical functions.  Use a `Num::Grad::Variable` with a `Num::Grad::Context`
to easily compute these derivatives.

```crystal
ctx = Num::Grad::Context(Tensor(Float64, CPU(Float64))).new

x = ctx.variable([3.0].to_tensor)
y = ctx.variable([2.0].to_tensor)

# f(x) = x ** y
f = x ** y
puts f # => [9]

f.backprop

# df/dx = y * x = 6.0
puts x.grad # => [6.0]
```

`Num::NN` contains an extension to `Num::Grad` that provides an easy-to-use
interface to assist in creating neural networks.  Designing and creating
a network is simple using Crystal's block syntax.

```crystal
ctx = Num::Grad::Context(Tensor(Float64, CPU(Float64))).new

x_train = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]].to_tensor
y_train = [[0.0], [1.0], [1.0], [0.0]].to_tensor

x = ctx.variable(x_train)

net = Num::NN::Network.new(ctx) do
  input [2]
  # A basic network with a single hidden layer using
  # a ReLU activation function
  linear 3
  relu
  linear 1

  # SGD Optimizer
  sgd 0.7

  # Sigmoid Cross Entropy to calculate loss
  sigmoid_cross_entropy_loss
end

500.times do |epoch|
  y_pred = net.forward(x)
  loss = net.loss(y_pred, y_train)
  puts "Epoch: #{epoch} - Loss #{loss}"
  loss.backprop
  net.optimizer.update
end

# Clip results to make a prediction
puts net.forward(x).value.map { |el| el > 0 ? 1 : 0}

# [[0],
#  [1],
#  [1],
#  [0]]
```

### Advanced Features

`num.cr` supports several advanced features for higher-performance numerical computing:

1. **Advanced Indexing**: Retrieve elements via boolean masks (`t[t > 2.0]`) or integer index arrays (`t[Tensor.from_array([0, 2])]`), and assign values back to them (`t[mask] = 99.0` or `t[mask] = replacement_tensor`).
2. **C vs. Fortran Contiguous Layouts**: Query memory ordering contiguity using `is_c_contiguous?` (Row-Major) and `is_f_contiguous?` (Column-Major).
3. **Array Manipulation APIs**: Horizontal and vertical stacking (`Tensor.hstack`, `Tensor.vstack`), cyclic shifts (`roll`), and reversals along dimensions (`flip`).
4. **Polynomial Mathematics**: Construct polynomials using `Num::Polynomial.new([c0, c1, ...])` and evaluate, add, multiply, or compute derivatives.
5. **Tensor Contractions**: Outer products (`Tensor.outer`) and vector cross products (`Tensor.cross`).
6. **Binary Serialization & Text I/O**: Fast serialization of tensors to disk (`save` and `load`) and CSV/TSV table loading (`Tensor.loadtxt`).
7. **Ufunc Reduction Helpers**: Running accumulation operators (`accumulate(:add)`) and pairwise broadcasting outer applications (`Tensor.ufunc_outer(a, b, :multiply)`).
8. **Masked Tensors**: Wrapper to track invalid/ignored elements during calculations (`Num::MaskedTensor`).
9. **Sparse Tensors**: Support for Coordinate format sparse matrices (`Num::SparseCOOTensor`) and sparse-vector matrix multiplication.

Review the documentation for full implementation details, and if something is missing,
open an issue to add it!

