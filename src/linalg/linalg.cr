# Copyright (c) 2021 Crystal Data Contributors
#
# MIT License
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to
# the following conditions:

# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
# LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
# WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

require "./extension"
require "./work"
require "complex"

# Re-implementing basic LAPACK bindings to avoid broken macros in extension.cr
lib LibLapack
  fun dgesv = dgesv_(n : Int32*, nrhs : Int32*, a : Float64*, lda : Int32*, ipiv : Int32*, b : Float64*, ldb : Int32*, info : Int32*) : Int32
  fun dgeev = dgeev_(jobvl : UInt8*, jobvr : Int8*, n : Int32*, a : Float64*, lda : Int32*, wr : Float64*, wi : Float64*, vl : Float64*, ldvl : Int32*, vr : Float64*, ldvr : Int32*, work : Float64*, lwork : Int32*, info : Int32*) : Int32
  fun dgetrf = dgetrf_(m : Int32*, n : Int32*, a : Float64*, lda : Int32*, ipiv : Int32*, info : Int32*) : Int32
  fun dgetri = dgetri_(n : Int32*, a : Float64*, lda : Int32*, ipiv : Int32*, work : Float64*, lwork : Int32*, info : Int32*) : Int32
end

class Tensor(T, S)
  def matmul(other : Tensor(T, S))
    self.assert_is_matrix
    other.assert_is_matrix
    if self.shape[1] != other.shape[0]
      raise "Matrix dimensions must agree"
    end
    m = self.shape[0].to_i32
    k = self.shape[1].to_i32
    n = other.shape[1].to_i32
    res = Tensor(T, S).new([m.to_i, n.to_i])
    
    {% if T == Float32 %}
      LibCblas.sgemm(LibCblas::ROW_MAJOR, LibCblas::CblasTranspose::CblasNoTrans, LibCblas::CblasTranspose::CblasNoTrans, m, n, k, 1.0_f32, self.get_offset_ptr_c, k, other.get_offset_ptr_c, n, 0.0_f32, res.get_offset_ptr_c, n)
    {% elsif T == Float64 %}
      LibCblas.dgemm(LibCblas::ROW_MAJOR, LibCblas::CblasTranspose::CblasNoTrans, LibCblas::CblasTranspose::CblasNoTrans, m, n, k, 1.0_f64, self.get_offset_ptr_c, k, other.get_offset_ptr_c, n, 0.0_f64, res.get_offset_ptr_c, n)
    {% end %}
    res
  end

  def solve(x : Tensor(T, S))
    self.assert_square_matrix
    a = dup(Num::ColMajor)
    x = x.dup(Num::ColMajor)
    n = a.shape[0].to_i32
    m = (x.rank > 1 ? x.shape[1] : 1).to_i32
    ipiv = Pointer(Int32).malloc(n)
    info = 0
    LibLapack.dgesv(pointerof(n), pointerof(m), a.to_unsafe, pointerof(n), ipiv, x.to_unsafe, pointerof(n), pointerof(info))
    raise "LAPACK dgesv returned #{info}" if info != 0
    x
  end

  def inv
    self.assert_square_matrix
    a = dup(Num::ColMajor)
    n = a.shape[0].to_i32
    ipiv = Pointer(Int32).malloc(n)
    info = 0
    LibLapack.dgetrf(pointerof(n), pointerof(n), a.to_unsafe, pointerof(n), ipiv, pointerof(info))
    raise "LAPACK dgetrf returned #{info}" if info != 0
    
    lwork = n
    work = Pointer(Float64).malloc(lwork)
    LibLapack.dgetri(pointerof(n), a.to_unsafe, pointerof(n), ipiv, work, pointerof(lwork), pointerof(info))
    raise "LAPACK dgetri returned #{info}" if info != 0
    a
  end

  def eigvals
    self.assert_square_matrix
    a = self.dup(Num::ColMajor)
    n = a.shape[0].to_i32
    wr = Array(Float64).new(n, 0.0)
    wi = Array(Float64).new(n, 0.0)
    
    jobvl = 'N'.ord.to_u8
    jobvr = 'N'.ord.to_u8
    lda = n
    ldvl = 1
    ldvr = 1
    vl_dummy = 0.0
    vr_dummy = 0.0
    lwork = 4 * n
    work = Pointer(Float64).malloc(lwork)
    info = 0
    
    LibLapack.dgeev(pointerof(jobvl), pointerof(jobvr), pointerof(n), a.to_unsafe, pointerof(lda),
      wr.to_unsafe, wi.to_unsafe, pointerof(vl_dummy), pointerof(ldvl), pointerof(vr_dummy), pointerof(ldvr), 
      work, pointerof(lwork), pointerof(info))
    
    raise "LAPACK dgeev returned #{info}" if info != 0
    
    res = Array(Complex).new(n)
    n.times do |i|
      res << Complex.new(wr[i], wi[i])
    end
    res
  end

  # Keep other methods or simplify
  def svd
    # Fallback to original if needed, but let's just use what works for cryspace
    raise "SVD not yet optimized in fork"
  end
  
  def is_f_contiguous; @flags.fortran?; end
  def is_c_contiguous; @flags.contiguous?; end
  def assert_square_matrix; raise "Input must be a square matrix" unless self.rank == 2 && self.shape[0] == self.shape[1]; end
  def assert_is_vector; raise "Inputs must be vectors" unless self.rank == 1; end
  def assert_is_matrix; raise "Input must be a matrix" unless self.rank == 2; end
end
