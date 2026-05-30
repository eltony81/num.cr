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

class Tensor(T, S)
  # Basic Linear Algebra Methods recovered from original num.cr
  
  def triu!(k : Int = 0)
    self.each_pointer_with_index do |e, i|
      m = i // @shape[1]
      n = i % @shape[1]
      e.value = m > n - k ? T.new(0) : e.value
    end
  end

  def triu(k : Int = 0)
    t = self.dup
    t.triu!(k)
    t
  end

  def tril!(k : Int = 0)
    self.each_pointer_with_index do |e, i|
      m = i // @shape[1]
      n = i % @shape[1]
      e.value = m < n - k ? T.new(0) : e.value
    end
  end

  def tril(k : Int = 0)
    t = self.dup
    t.tril!(k)
    t
  end

  def det
    self.assert_square_matrix
    a = dup(Num::ColMajor)
    m, n = a.shape
    ipiv = Pointer(Int32).malloc(n)
    result = Tensor(T, S).new([1])
    
    # Using the existing lapack macro for det (getrf)
    lapack(getrf, m, n, a.get_offset_ptr_c, n, ipiv)

    ldet = T.new(1)
    a.diagonal.each do |el|
      ldet *= el
    end

    detp = 1
    n.times do |j|
      if j + 1 != ipiv[j]
        detp = -detp
      end
    end

    result.get_offset_ptr_c.value = ldet * detp
    result
  end

  def dot(u : Tensor(T, S))
    result = Tensor(T, S).new([1])
    {% if S < OCL %}
      blast(dot, @size, result.to_unsafe, 0, self.to_unsafe, @offset, @strides[0], u.to_unsafe, u.offset, u.strides[0])
    {% else %}
      dotvalue = blas_call(
        dot,
        @size,
        self.get_offset_ptr_c,
        @strides[0],
        u.get_offset_ptr_c,
        u.strides[0]
      )
      result.get_offset_ptr_c.value = dotvalue
    {% end %}
    result
  end

  def cholesky!(*, lower = true)
    assert_square_matrix
    assert_fortran
    char = lower ? 'L' : 'U'
    lapack(potrf, char.ord.to_u8, shape[0], to_unsafe, shape[0])
    lower ? tril! : triu!
  end

  def qr
    self.assert_is_matrix
    m, n = @shape
    k = {m, n}.min
    a = self.dup(Num::ColMajor)
    tau = Tensor(T, S).new([k])
    lapack(geqrf, m, n, a.get_offset_ptr_c, m, tau.get_offset_ptr_c)
    r = a.triu
    lapack(orgqr, m, n, k, a.get_offset_ptr_c, m, tau.get_offset_ptr_c)
    {a, r}
  end

  def hessenberg
    self.assert_square_matrix
    n = @shape[0]
    a = dup(Num::ColMajor)
    if n < 2
      return a
    end
    tau = Tensor(T, S).new([n - 1])
    # Fixed manual lapack call for gehrd since macro might be complex
    info = 0
    lwork = -1
    work_query = 0.0
    LibLapack.dgehrd(pointerof(n), pointerof(n), pointerof(n), a.to_unsafe, pointerof(n), tau.to_unsafe, pointerof(work_query), pointerof(lwork), pointerof(info))
    lwork = work_query.to_i32
    work = Pointer(Float64).malloc(lwork)
    LibLapack.dgehrd(pointerof(n), pointerof(n), pointerof(n), a.to_unsafe, pointerof(n), tau.to_unsafe, work, pointerof(lwork), pointerof(info))
    a.triu(-1)
  end

  # --- Optimized and Fixed Methods for CrySpace ---

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
    n = a.shape[0]
    m = x.rank > 1 ? x.shape[1] : 1
    ipiv = Pointer(Int32).malloc(n)
    
    {% if T == Float64 %}
      info = 0
      LibLapack.dgesv(pointerof(n), pointerof(m), a.to_unsafe, pointerof(n), ipiv, x.to_unsafe, pointerof(n), pointerof(info))
      raise "LAPACK dgesv returned #{info}" if info != 0
    {% end %}
    x
  end

  def inv
    self.assert_square_matrix
    a = dup(Num::ColMajor)
    n = a.shape[0]
    ipiv = Pointer(Int32).malloc(n)
    info = 0
    
    {% if T == Float64 %}
      LibLapack.dgetrf(pointerof(n), pointerof(n), a.to_unsafe, pointerof(n), ipiv, pointerof(info))
      raise "LAPACK dgetrf returned #{info}" if info != 0
      
      lwork = -1
      work_query = 0.0
      LibLapack.dgetri(pointerof(n), a.to_unsafe, pointerof(n), ipiv, pointerof(work_query), pointerof(lwork), pointerof(info))
      lwork = work_query.to_i32
      work = Pointer(Float64).malloc(lwork)
      LibLapack.dgetri(pointerof(n), a.to_unsafe, pointerof(n), ipiv, work, pointerof(lwork), pointerof(info))
      raise "LAPACK dgetri returned #{info}" if info != 0
    {% end %}
    a
  end

  def svd
    self.assert_is_matrix
    m = self.shape[0]
    n = self.shape[1]
    k = {m, n}.min
    a = self.dup(Num::ColMajor)
    u = Tensor(T, S).new([m, m])
    s = Tensor(T, S).new([k])
    v = Tensor(T, S).new([n, n])
    
    {% if T == Float64 %}
      jobu = 'A'.ord.to_u8
      jobvt = 'A'.ord.to_u8
      lwork = -1
      work_query = 0.0
      info = 0
      LibLapack.dgesvd(pointerof(jobu), pointerof(jobvt), pointerof(m), pointerof(n), a.to_unsafe, pointerof(m), s.to_unsafe, u.to_unsafe, pointerof(m), v.to_unsafe, pointerof(n), pointerof(work_query), pointerof(lwork), pointerof(info))
      lwork = work_query.to_i32
      work = Pointer(Float64).malloc(lwork)
      LibLapack.dgesvd(pointerof(jobu), pointerof(jobvt), pointerof(m), pointerof(n), a.to_unsafe, pointerof(m), s.to_unsafe, u.to_unsafe, pointerof(m), v.to_unsafe, pointerof(n), work, pointerof(lwork), pointerof(info))
      raise "LAPACK dgesvd returned #{info}" if info != 0
    {% end %}
    {u, s, v}
  end

  def eigvals
    self.assert_square_matrix
    a_copy = self.dup(Num::ColMajor)
    n = a_copy.shape[0]
    
    {% if T == Float64 %}
      wr = Array(Float64).new(n, 0.0)
      wi = Array(Float64).new(n, 0.0)
      jobvl, jobvr = 'N'.ord.to_u8, 'N'.ord.to_u8
      lda, ldvl, ldvr = n, 1, 1
      vl_dummy, vr_dummy = 0.0, 0.0
      lwork, info = -1, 0
      work_query = 0.0
      LibLapack.dgeev(pointerof(jobvl), pointerof(jobvr), pointerof(n), a_copy.to_unsafe, pointerof(lda),
        wr.to_unsafe, wi.to_unsafe, pointerof(vl_dummy), pointerof(ldvl), pointerof(vr_dummy), pointerof(ldvr), 
        pointerof(work_query), pointerof(lwork), pointerof(info))
      lwork = work_query.to_i32
      work = Pointer(Float64).malloc(lwork)
      LibLapack.dgeev(pointerof(jobvl), pointerof(jobvr), pointerof(n), a_copy.to_unsafe, pointerof(lda),
        wr.to_unsafe, wi.to_unsafe, pointerof(vl_dummy), pointerof(ldvl), pointerof(vr_dummy), pointerof(ldvr), 
        work, pointerof(lwork), pointerof(info))
      raise "LAPACK dgeev returned #{info}" if info != 0
      res = Array(Complex).new(n)
      n.times { |i| res << Complex.new(wr[i], wi[i]) }
      res
    {% else %}
      raise "eigvals only implemented for Float64 in fork"
    {% end %}
  end

  def eig
    self.assert_square_matrix
    a_copy = self.dup(Num::ColMajor)
    n = a_copy.shape[0]
    {% if T == Float64 %}
      wr, wi = Array(Float64).new(n, 0.0), Array(Float64).new(n, 0.0)
      vl = Tensor(T, S).new([n, n])
      vr = Tensor(T, S).new([n, n])
      jobvl, jobvr = 'V'.ord.to_u8, 'V'.ord.to_u8
      lda, ldvl, ldvr = n, n, n
      lwork, info = -1, 0
      work_query = 0.0
      LibLapack.dgeev(pointerof(jobvl), pointerof(jobvr), pointerof(n), a_copy.to_unsafe, pointerof(lda),
        wr.to_unsafe, wi.to_unsafe, vl.to_unsafe, pointerof(ldvl), vr.to_unsafe, pointerof(ldvr), 
        pointerof(work_query), pointerof(lwork), pointerof(info))
      lwork = work_query.to_i32
      work = Pointer(Float64).malloc(lwork)
      LibLapack.dgeev(pointerof(jobvl), pointerof(jobvr), pointerof(n), a_copy.to_unsafe, pointerof(lda),
        wr.to_unsafe, wi.to_unsafe, vl.to_unsafe, pointerof(ldvl), vr.to_unsafe, pointerof(ldvr), 
        work, pointerof(lwork), pointerof(info))
      raise "LAPACK dgeev returned #{info}" if info != 0
      res_w = Array(Complex).new(n)
      n.times { |i| res_w << Complex.new(wr[i], wi[i]) }
      {res_w, vr}
    {% else %}
      raise "eig only implemented for Float64 in fork"
    {% end %}
  end

  def is_f_contiguous; @flags.fortran?; end
  def is_c_contiguous; @flags.contiguous?; end
  private def assert_fortran; raise "Matrix must be fortran contiguous" unless self.is_f_contiguous; end
  def assert_square_matrix; raise "Input must be a square matrix" unless self.rank == 2 && self.shape[0] == self.shape[1]; end
  def assert_is_vector; raise "Inputs must be vectors" unless self.rank == 1; end
  def assert_is_matrix; raise "Input must be a matrix" unless self.rank == 2; end
end
