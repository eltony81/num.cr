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

require "complex"

class Tensor(T, S)
  def triu!(k : Int = 0)
    m, n = @shape
    if self.is_c_contiguous
      ptr = self.to_unsafe
      m.times do |i|
        n.times do |j|
          if i > j - k
            ptr[i * n + j] = T.new(0)
          end
        end
      end
    else
      # Fallback to slower but safe method for non-contiguous
      self.each_pointer_with_index do |e, i|
        r = i // n
        c = i % n
        e.value = r > c - k ? T.new(0) : e.value
      end
    end
    self
  end

  def triu(k : Int = 0)
    t = self.dup
    t.triu!(k)
    t
  end

  def tril!(k : Int = 0)
    m, n = @shape
    if self.is_c_contiguous
      ptr = self.to_unsafe
      m.times do |i|
        n.times do |j|
          if i < j - k
            ptr[i * n + j] = T.new(0)
          end
        end
      end
    else
      self.each_pointer_with_index do |e, i|
        r = i // n
        c = i % n
        e.value = r < c - k ? T.new(0) : e.value
      end
    end
    self
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
    lapack(getrf, m, n, a.get_offset_ptr_c, n, ipiv)
    ldet = T.new(1)
    a.diagonal.each { |el| ldet *= el }
    detp = 1
    n.times { |j| detp = -detp if j + 1 != ipiv[j] }
    result.get_offset_ptr_c.value = ldet * detp
    result
  end

  def dot(u : Tensor(T, S))
    result = Tensor(T, S).new([1])
    {% if S < OCL %}
      blast(dot, @size, result.to_unsafe, 0, self.to_unsafe, @offset, @strides[0], u.to_unsafe, u.offset, u.strides[0])
    {% else %}
      dotvalue = blas_call(dot, @size, self.get_offset_ptr_c, @strides[0], u.get_offset_ptr_c, u.strides[0])
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
    lapack(geqrf, m, n, a.get_offset_ptr_c, m, tau.get_offset_ptr_c, worksize: n)
    r = a.triu
    lapack(orgqr, m, n, k, a.get_offset_ptr_c, m, tau.get_offset_ptr_c, worksize: n)
    {a, r}
  end

  def hessenberg
    self.assert_square_matrix
    n = @shape[0]
    a = dup(Num::ColMajor)
    return {a, a} if n < 2
    tau = Tensor(T, S).new([n - 1])
    lapack(gehrd, n, 1, n, a.get_offset_ptr_c, n, tau.get_offset_ptr_c, worksize: n)
    h = a.triu(-1)
    lapack(orghr, n, 1, n, a.get_offset_ptr_c, n, tau.get_offset_ptr_c, worksize: n)
    {h, a}
  end

  def matmul(other : Tensor(T, S), output : Tensor(T, S)? = nil)
    self.assert_is_matrix
    other.assert_is_matrix
    raise "Matrix dimensions must agree" if self.shape[1] != other.shape[0]
    m, k, n = self.shape[0], self.shape[1], other.shape[1]
    res = output || Tensor(T, S).new([m, n])
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
    {% if T == Float64 %}
      info = 0
      LibLapack.dgesv(pointerof(n), pointerof(m), a.to_unsafe, pointerof(n), ipiv, x.to_unsafe, pointerof(n), pointerof(info))
      raise "LAPACK dgesv returned #{info}" if info != 0
    {% elsif T == Float32 %}
      info = 0
      LibLapack.sgesv(pointerof(n), pointerof(m), a.to_unsafe, pointerof(n), ipiv, x.to_unsafe, pointerof(n), pointerof(info))
      raise "LAPACK sgesv returned #{info}" if info != 0
    {% end %}
    x
  end

  def inv
    self.assert_square_matrix
    a = dup(Num::ColMajor)
    n = a.shape[0].to_i32
    ipiv = Pointer(Int32).malloc(n)
    {% if T == Float64 %}
      info = 0
      LibLapack.dgetrf(pointerof(n), pointerof(n), a.to_unsafe, pointerof(n), ipiv, pointerof(info))
      raise "LAPACK dgetrf returned #{info}" if info != 0
      lwork = -1
      work_query = 0.0
      LibLapack.dgetri(pointerof(n), a.to_unsafe, pointerof(n), ipiv, pointerof(work_query), pointerof(lwork), pointerof(info))
      lwork = work_query.to_i32
      work = Pointer(Float64).malloc(lwork)
      LibLapack.dgetri(pointerof(n), a.to_unsafe, pointerof(n), ipiv, work, pointerof(lwork), pointerof(info))
      raise "LAPACK dgetri returned #{info}" if info != 0
    {% elsif T == Float32 %}
      info = 0
      LibLapack.sgetrf(pointerof(n), pointerof(n), a.to_unsafe, pointerof(n), ipiv, pointerof(info))
      raise "LAPACK sgetrf returned #{info}" if info != 0
      lwork = -1
      work_query = 0.0_f32
      LibLapack.sgetri(pointerof(n), a.to_unsafe, pointerof(n), ipiv, pointerof(work_query), pointerof(lwork), pointerof(info))
      lwork = work_query.to_i32
      work = Pointer(Float32).malloc(lwork)
      LibLapack.sgetri(pointerof(n), a.to_unsafe, pointerof(n), ipiv, work, pointerof(lwork), pointerof(info))
      raise "LAPACK sgetri returned #{info}" if info != 0
    {% end %}
    a
  end

  def svd
    self.assert_is_matrix
    m = self.shape[0].to_i32
    n = self.shape[1].to_i32
    k = {m, n}.min
    a = self.dup(Num::ColMajor)
    u = Tensor(T, S).new([m.to_i, m.to_i])
    s = Tensor(T, S).new([k.to_i])
    v = Tensor(T, S).new([n.to_i, n.to_i])
    jobu = 'A'.ord.to_u8
    jobvt = 'A'.ord.to_u8
    {% if T == Float64 %}
      lwork = -1
      work_query = 0.0
      info = 0
      LibLapack.dgesvd(pointerof(jobu), pointerof(jobvt), pointerof(m), pointerof(n), a.to_unsafe, pointerof(m), s.to_unsafe, u.to_unsafe, pointerof(m), v.to_unsafe, pointerof(n), pointerof(work_query), pointerof(lwork), pointerof(info))
      lwork = work_query.to_i32
      work = Pointer(Float64).malloc(lwork)
      LibLapack.dgesvd(pointerof(jobu), pointerof(jobvt), pointerof(m), pointerof(n), a.to_unsafe, pointerof(m), s.to_unsafe, u.to_unsafe, pointerof(m), v.to_unsafe, pointerof(n), work, pointerof(lwork), pointerof(info))
      raise "LAPACK dgesvd returned #{info}" if info != 0
    {% elsif T == Float32 %}
      lwork = -1
      work_query = 0.0_f32
      info = 0
      LibLapack.sgesvd(pointerof(jobu), pointerof(jobvt), pointerof(m), pointerof(n), a.to_unsafe, pointerof(m), s.to_unsafe, u.to_unsafe, pointerof(m), v.to_unsafe, pointerof(n), pointerof(work_query), pointerof(lwork), pointerof(info))
      lwork = work_query.to_i32
      work = Pointer(Float32).malloc(lwork)
      LibLapack.sgesvd(pointerof(jobu), pointerof(jobvt), pointerof(m), pointerof(n), a.to_unsafe, pointerof(m), s.to_unsafe, u.to_unsafe, pointerof(m), v.to_unsafe, pointerof(n), work, pointerof(lwork), pointerof(info))
      raise "LAPACK sgesvd returned #{info}" if info != 0
    {% end %}
    {u.transpose, s, v.transpose}
  end

  def eigh
    self.assert_square_matrix
    a = dup(Num::ColMajor)
    n = a.shape[0]
    w = Tensor(T, S).new([n])
    lapack(syev, 'V'.ord.to_u8, 'L'.ord.to_u8, n, a.get_offset_ptr_c, n, w.get_offset_ptr_c, worksize: 3 * n - 1)
    {w, a}
  end

  def eigvalsh
    self.assert_square_matrix
    a = dup(Num::ColMajor)
    n = a.shape[0]
    w = Tensor(T, S).new([n])
    lapack(syev, 'N'.ord.to_u8, 'L'.ord.to_u8, n, a.get_offset_ptr_c, n, w.get_offset_ptr_c, worksize: 3 * n - 1)
    w
  end

  def eigvals
    self.assert_square_matrix
    a = self.dup(Num::ColMajor)
    n = a.shape[0]
    {% if T == Float32 || T == Float64 %}
      wr, wi = Tensor(T, S).new([n]), Tensor(T, S).new([n])
      vl_dummy, vr_dummy = Tensor(T, S).new([1, 1]), Tensor(T, S).new([1, 1])
      lapack(geev, 'N'.ord.to_u8, 'N'.ord.to_u8, n, a.get_offset_ptr_c, n, wr.get_offset_ptr_c,
        wi.get_offset_ptr_c, vl_dummy.get_offset_ptr_c, 1, vr_dummy.get_offset_ptr_c, 1, worksize: 4 * n)
      wr
    {% else %}
      raise "eigvals not implemented"
    {% end %}
  end

  def eig
    self.assert_square_matrix
    a = dup(Num::ColMajor)
    n = a.shape[0]
    {% if T == Float32 || T == Float64 %}
      wr, wi = Tensor(T, S).new([n]), Tensor(T, S).new([n])
      vl, vr = Tensor(T, S).new([n, n]), Tensor(T, S).new([n, n])
      lapack(geev, 'V'.ord.to_u8, 'V'.ord.to_u8, n, a.get_offset_ptr_c, n, wr.get_offset_ptr_c,
        wi.get_offset_ptr_c, vl.get_offset_ptr_c, n, vr.get_offset_ptr_c, n, worksize: 4 * n)
      {wr, vl}
    {% else %}
      raise "eig not implemented"
    {% end %}
  end

  def eigvals_c
    self.assert_square_matrix
    a = self.dup(Num::ColMajor)
    n = a.shape[0].to_i32
    wr = Pointer(T).malloc(n)
    wi = Pointer(T).malloc(n)
    jobvl, jobvr = 'N'.ord.to_u8, 'N'.ord.to_u8
    lda, ldvl, ldvr = n, 1, 1
    vl_dummy, vr_dummy = T.new(0), T.new(0)
    lwork, info = -1, 0
    work_query = T.new(0)
    {% if T == Float64 %}
      LibLapack.dgeev(pointerof(jobvl), pointerof(jobvr), pointerof(n), a.to_unsafe, pointerof(lda),
        wr, wi, pointerof(vl_dummy), pointerof(ldvl), pointerof(vr_dummy), pointerof(ldvr), 
        pointerof(work_query), pointerof(lwork), pointerof(info))
      lwork = work_query.to_i32
      work = Pointer(Float64).malloc(lwork)
      LibLapack.dgeev(pointerof(jobvl), pointerof(jobvr), pointerof(n), a.to_unsafe, pointerof(lda),
        wr, wi, pointerof(vl_dummy), pointerof(ldvl), pointerof(vr_dummy), pointerof(ldvr), 
        work, pointerof(lwork), pointerof(info))
    {% elsif T == Float32 %}
      LibLapack.sgeev(pointerof(jobvl), pointerof(jobvr), pointerof(n), a.to_unsafe, pointerof(lda),
        wr, wi, pointerof(vl_dummy), pointerof(ldvl), pointerof(vr_dummy), pointerof(ldvr), 
        pointerof(work_query), pointerof(lwork), pointerof(info))
      lwork = work_query.to_i32
      work = Pointer(Float32).malloc(lwork)
      LibLapack.sgeev(pointerof(jobvl), pointerof(jobvr), pointerof(n), a.to_unsafe, pointerof(lda),
        wr, wi, pointerof(vl_dummy), pointerof(ldvl), pointerof(vr_dummy), pointerof(ldvr), 
        work, pointerof(lwork), pointerof(info))
    {% end %}
    raise "LAPACK geev returned #{info}" if info != 0
    res = Tensor(Complex, CPU(Complex)).new([n.to_i])
    n.times { |i| res.to_unsafe[i] = Complex.new(wr[i].to_f64, wi[i].to_f64) }
    res
  end

  def eig_c
    self.assert_square_matrix
    a = self.dup(Num::ColMajor)
    n = a.shape[0].to_i32
    wr, wi = Pointer(T).malloc(n), Pointer(T).malloc(n)
    vl = Tensor(T, S).new([n.to_i, n.to_i])
    vr = Tensor(T, S).new([n.to_i, n.to_i])
    jobvl, jobvr = 'V'.ord.to_u8, 'V'.ord.to_u8
    lda, ldvl, ldvr = n, n, n
    lwork, info = -1, 0
    work_query = T.new(0)
    {% if T == Float64 %}
      LibLapack.dgeev(pointerof(jobvl), pointerof(jobvr), pointerof(n), a.to_unsafe, pointerof(lda),
        wr, wi, vl.to_unsafe, pointerof(ldvl), vr.to_unsafe, pointerof(ldvr), 
        pointerof(work_query), pointerof(lwork), pointerof(info))
      lwork = work_query.to_i32
      work = Pointer(Float64).malloc(lwork)
      LibLapack.dgeev(pointerof(jobvl), pointerof(jobvr), pointerof(n), a.to_unsafe, pointerof(lda),
        wr, wi, vl.to_unsafe, pointerof(ldvl), vr.to_unsafe, pointerof(ldvr), 
        work, pointerof(lwork), pointerof(info))
    {% elsif T == Float32 %}
      LibLapack.sgeev(pointerof(jobvl), pointerof(jobvr), pointerof(n), a.to_unsafe, pointerof(lda),
        wr, wi, vl.to_unsafe, pointerof(ldvl), vr.to_unsafe, pointerof(ldvr), 
        pointerof(work_query), pointerof(lwork), pointerof(info))
      lwork = work_query.to_i32
      work = Pointer(Float32).malloc(lwork)
      LibLapack.sgeev(pointerof(jobvl), pointerof(jobvr), pointerof(n), a.to_unsafe, pointerof(lda),
        wr, wi, vl.to_unsafe, pointerof(ldvl), vr.to_unsafe, pointerof(ldvr), 
        work, pointerof(lwork), pointerof(info))
    {% end %}
    raise "LAPACK geev returned #{info}" if info != 0
    res_w = Tensor(Complex, CPU(Complex)).new([n.to_i])
    n.times { |i| res_w.to_unsafe[i] = Complex.new(wr[i].to_f64, wi[i].to_f64) }
    {res_w, vr}
  end

  def norm(order = 'F')
    self.assert_is_matrix
    a = self.dup(Num::ColMajor)
    m, n = a.shape
    result = Tensor(T, S).new([1])
    worksize = order == 'I' ? m : 0
    r = lapack_util(lange, worksize, order.ord.to_u8, m, n, tensor(a.get_offset_ptr_c), m)
    result.get_offset_ptr_c.value = r
    result
  end

  def is_f_contiguous; @flags.fortran?; end
  def is_c_contiguous; @flags.contiguous?; end
  private def assert_fortran; raise "Matrix must be fortran contiguous" unless self.is_f_contiguous; end
  def assert_square_matrix; raise "Matrix must be square" unless rank == 2 && @shape[0] == @shape[1]; end
  def assert_is_vector; raise "Inputs must be vectors" unless self.rank == 1; end
  def assert_is_matrix; raise "Input must be a matrix" unless self.rank == 2; end
end
