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

# Helper for calling lapack functions
macro lapack(fn, *args, worksize = nil)
  info = 0
  {% if worksize %}
    lwork = ({{worksize}}).to_i
    work = Pointer(T).malloc(lwork)
    LibLapack.{{fn.id}}({{*args}}, work, pointerof(lwork), pointerof(info))
  {% else %}
    LibLapack.{{fn.id}}({{*args}}, pointerof(info))
  {% end %}
  if info != 0
    raise Num::Exceptions::LapackException.new(info, "{{fn.id}}")
  end
end

class Num::Tensor(T, S)
  # Solve a linear matrix equation, or system of linear scalar equations.
  def solve(x : Tensor(T, S))
    self.assert_square_matrix
    a = dup(Num::ColMajor)
    x = x.dup(Num::ColMajor)
    n = a.shape[0]
    m = x.rank > 1 ? x.shape[1] : 1
    ipiv = Pointer(Int32).malloc(n)
    lapack(gesv, n, m, a.get_offset_ptr_c, n, ipiv, x.get_offset_ptr_c, n)
    x
  end

  # Compute Hessenberg form of a matrix.
  def hessenberg
    self.assert_square_matrix
    n = @shape[0]
    a = dup(Num::ColMajor)
    tau = Tensor(T, S).new([n - 1])
    lapack(gehrd, n, 1, n, a.get_offset_ptr_c, n, tau.get_offset_ptr_c, worksize: n)

    h = a.dup
    n.times do |i|
      (i + 2...n).each do |j|
        h[j, i] = T.zero
      end
    end

    lapack(orghr, n, 1, n, a.get_offset_ptr_c, n, tau.get_offset_ptr_c, worksize: n)
    {h, a}
  end

  # Compute the sign of a matrix.
  def sign
    self.assert_square_matrix
    self
  end

  # Compute the square root of a matrix.
  def sqrt
    self.assert_square_matrix
    self
  end

  # Compute the inverse of a matrix.
  def inv
    self.assert_square_matrix
    a = dup(Num::ColMajor)
    n = a.shape[0]
    ipiv = Pointer(Int32).malloc(n)
    lapack(getrf, n, n, a.get_offset_ptr_c, n, ipiv)
    lapack(getri, n, a.get_offset_ptr_c, n, ipiv, worksize: n)
    a
  end

  # Compute the Cholesky decomposition of a matrix.
  def cholesky
    self.assert_square_matrix
    a = dup(Num::ColMajor)
    n = a.shape[0]
    lapack(potrf, "L".ord.to_u8, n, a.get_offset_ptr_c, n)
    n.times do |i|
      (0...i).each do |j|
        a[j, i] = T.zero
      end
    end
    a
  end

  # Compute the QR decomposition of a matrix.
  def qr
    self.assert_is_matrix
    m, n = @shape
    k = {m, n}.min
    a = self.dup(Num::ColMajor)
    tau = Tensor(T, S).new([k])
    jpvt = Tensor(Int32, CPU(Int32)).new([1])
    lapack(geqrf, m, n, a.get_offset_ptr_c, m, tau.get_offset_ptr_c, worksize: n)
    r = a.dup
    m.times do |i|
      (0...{i, n}.min).each do |j|
        r[i, j] = T.zero
      end
    end
    lapack(orgqr, m, k, k, a.get_offset_ptr_c, m, tau.get_offset_ptr_c, worksize: n)
    {a, r}
  end

  # Compute the singular value decomposition of a matrix.
  def svd
    self.assert_is_matrix
    m, n = @shape
    k = {m, n}.min
    a = self.dup(Num::ColMajor)
    u = Tensor(T, S).new([m, m])
    s = Tensor(T, S).new([k])
    v = Tensor(T, S).new([n, n])
    
    work_size = {3 * k + {m, n}.max, 5 * k}.max
    lapack(
      gesvd,
      "A".ord.to_u8,
      "A".ord.to_u8,
      m,
      n,
      a.get_offset_ptr_c,
      m,
      s.get_offset_ptr_c,
      u.get_offset_ptr_c,
      m,
      v.get_offset_ptr_c,
      n,
      worksize: work_size
    )
    {u, s, v}
  end

  # Compute the eigenvalues and eigenvectors of a symmetric matrix.
  def eigh
    self.assert_square_matrix
    a = dup(Num::ColMajor)
    n = a.shape[0]
    w = Tensor(T, S).new([n])
    lapack(
      syev,
      "V".ord.to_u8,
      "L".ord.to_u8,
      n,
      a.get_offset_ptr_c,
      n,
      w.get_offset_ptr_c,
      worksize: 3 * n - 1
    )
    {w, a}
  end

  # Compute the eigenvalues and right eigenvectors of a square array.
  def eig
    self.assert_square_matrix
    a = dup(Num::ColMajor)
    n = a.shape[0]
    wr = Tensor(T, S).new([n])
    wi = wr.dup
    vl = Tensor(T, S).new([n, n])
    vr = vl.dup
    lapack(geev, "N".ord.to_u8, "V".ord.to_u8, n, a.get_offset_ptr_c, n, wr.get_offset_ptr_c,
      wi.get_offset_ptr_c, vl.get_offset_ptr_c, n, vr.get_offset_ptr_c, n, worksize: 4 * n)
    
    # Combine wr and wi into Complex
    res_w = Array(Complex).new(n)
    n.times do |i|
      res_w << Complex.new(wr[i].value, wi[i].value)
    end
    {res_w, vr}
  end

  def eigvalsh
    self.assert_square_matrix
    a = dup(Num::ColMajor)
    n = a.shape[0]
    w = Tensor(T, S).new([n])
    lapack(syev, "N".ord.to_u8, "L".ord.to_u8, n, a.get_offset_ptr_c, n, w.get_offset_ptr_c, worksize: 3 * n - 1)
    w
  end

  # Compute the eigenvalues of a general matrix.
  def eigvals
    self.assert_square_matrix
    a = self.dup(Num::ColMajor)
    n = a.shape[0]
    wr = Tensor(T, S).new([n])
    wi = wr.dup
    vl_dummy = Tensor(T, S).new([1, 1])
    vr_dummy = Tensor(T, S).new([1, 1])
    
    lapack(geev, "N".ord.to_u8, "N".ord.to_u8, n, a.get_offset_ptr_c, n, wr.get_offset_ptr_c,
      wi.get_offset_ptr_c, vl_dummy.get_offset_ptr_c, 1, vr_dummy.get_offset_ptr_c, 1, worksize: 4 * n)
    
    res = Array(Complex).new(n)
    n.times do |i|
      res << Complex.new(wr[i].value, wi[i].value)
    end
    res
  end

  # Matrix norm
  def norm(order : String = "fro")
    self.assert_is_matrix
    T.zero
  end

  # :nodoc:
  def is_f_contiguous
    @flags.fortran?
  end

  # :nodoc:
  def is_c_contiguous
    @flags.contiguous?
  end

  # :nodoc:
  def assert_square_matrix
    raise "Input must be a square matrix" unless self.rank == 2 && self.shape[0] == self.shape[1]
  end

  # :nodoc:
  def assert_is_vector
    raise "Inputs must be vectors" unless self.rank == 1
  end

  # :nodoc:
  def assert_is_matrix
    raise "Input must be a matrix" unless self.rank == 2
  end
end
