require "../spec_helper"
require "spec"

describe "Nine Advanced Features" do
  # 1. Advanced Indexing
  it "performs advanced indexing (boolean indexing and integer indexing)" do
    # Boolean indexing
    a = Tensor.new([5]) { |i| i.to_f64 }
    mask = a > 2.0
    result = a[mask]
    result.size.should eq(2)
    result.to_unsafe[0].should eq(3.0)
    result.to_unsafe[1].should eq(4.0)

    # Boolean scalar assignment
    a[mask] = 99.0
    a.to_unsafe[3].should eq(99.0)
    a.to_unsafe[4].should eq(99.0)
    a.to_unsafe[0].should eq(0.0)

    # Boolean tensor assignment
    b = Tensor.new([5]) { |i| i.to_f64 }
    mask_b = b > 2.0
    replacements = Tensor.from_array([100.0, 101.0])
    b[mask_b] = replacements
    b.to_unsafe[3].should eq(100.0)
    b.to_unsafe[4].should eq(101.0)
    b.to_unsafe[0].should eq(0.0)

    # Integer indexing
    c = Tensor.new([4]) { |i| (i * 10).to_f64 }
    indices = Tensor.from_array([0, 2])
    c_indexed = c[indices]
    c_indexed.size.should eq(2)
    c_indexed.to_unsafe[0].should eq(0.0)
    c_indexed.to_unsafe[1].should eq(20.0)
  end

  # 2. Layout Contiguity
  it "detects C-contiguous and Fortran-contiguous memory layouts" do
    t_c = Tensor(Float64, CPU(Float64)).new([2, 3], Num::RowMajor)
    t_c.is_c_contiguous?.should be_true
    t_c.is_f_contiguous?.should be_false

    t_f = Tensor(Float64, CPU(Float64)).new([2, 3], Num::ColMajor)
    t_f.is_f_contiguous?.should be_true
    t_f.is_c_contiguous?.should be_false
  end

  # 3. Array Manipulation
  it "stacks, rolls, and flips tensors" do
    t1 = Tensor.new([2]) { |i| i.to_f64 }
    t2 = Tensor.new([2]) { |i| i.to_f64 + 2.0 }

    # hstack / vstack
    h = Tensor.hstack([t1, t2])
    h.size.should eq(4)
    h.to_unsafe[2].should eq(2.0)

    v = Tensor.vstack([t1, t2])
    v.shape.should eq([4])

    # roll
    r = Tensor.new([3]) { |i| i.to_f64 } # [0, 1, 2]
    rolled = r.roll(1)
    rolled.to_unsafe[0].should eq(2.0)
    rolled.to_unsafe[1].should eq(0.0)
    rolled.to_unsafe[2].should eq(1.0)

    # flip
    f = Tensor.new([3]) { |i| i.to_f64 } # [0, 1, 2]
    flipped = f.flip
    flipped.to_unsafe[0].should eq(2.0)
    flipped.to_unsafe[1].should eq(1.0)
    flipped.to_unsafe[2].should eq(0.0)
  end

  # 4. Polynomial Math
  it "evaluates and manipulates polynomials" do
    # p(x) = 1 + 2x + 3x^2
    poly = Num::Polynomial.new([1.0, 2.0, 3.0])
    poly.eval(2.0).should eq(1.0 + 4.0 + 12.0)

    # p(x) evaluation on tensor
    x_tensor = Tensor.new([2]) { |i| i.to_f64 + 1.0 } # [1.0, 2.0]
    poly_tensor = poly.eval(x_tensor)
    poly_tensor.to_unsafe[0].should eq(6.0) # 1 + 2(1) + 3(1) = 6
    poly_tensor.to_unsafe[1].should eq(17.0) # 1 + 2(2) + 3(4) = 17

    # Addition
    poly2 = Num::Polynomial.new([2.0, 3.0])
    added = poly + poly2
    added.coeffs.should eq([3.0, 5.0, 3.0])

    # Multiplication
    # (1 + 2x) * (2 + 3x) = 2 + 7x + 6x^2
    p_a = Num::Polynomial.new([1.0, 2.0])
    p_b = Num::Polynomial.new([2.0, 3.0])
    prod = p_a * p_b
    prod.coeffs.should eq([2.0, 7.0, 6.0])

    # Derivative
    # d/dx(1 + 2x + 3x^2) = 2 + 6x
    deriv = poly.derivative
    deriv.coeffs.should eq([2.0, 6.0])
  end

  # 5. Tensor Contractions
  it "computes outer and cross products" do
    a = Tensor.new([2]) { |i| i.to_f64 + 1.0 } # [1, 2]
    b = Tensor.new([3]) { |i| i.to_f64 + 3.0 } # [3, 4, 5]

    # outer
    out = Tensor.outer(a, b)
    out.shape.should eq([2, 3])
    out.to_unsafe[0].should eq(3.0)
    out.to_unsafe[1].should eq(4.0)
    out.to_unsafe[3].should eq(6.0) # 2 * 3

    # cross
    v1 = Tensor.from_array([1.0, 0.0, 0.0])
    v2 = Tensor.from_array([0.0, 1.0, 0.0])
    cross_prod = Tensor.cross(v1, v2)
    cross_prod.to_a.should eq([0.0, 0.0, 1.0])
  end

  # 6. Binary Serialization & Text I/O
  it "serializes tensors to binary and parses text databases" do
    filename_bin = "test_tensor.bin"
    filename_txt = "test_tensor.csv"

    # Binary Save/Load
    a = Tensor.new([2, 2]) { |i| i.to_f64 }
    a.save(filename_bin)
    
    loaded = Tensor(Float64, CPU(Float64)).load(filename_bin)
    loaded.shape.should eq(a.shape)
    loaded.to_unsafe[3].should eq(3.0)

    # Text I/O loadtxt
    File.write(filename_txt, "1.0, 2.0\n3.0, 4.0\n# comment line\n5.0, 6.0")
    loaded_txt = Tensor.loadtxt(filename_txt)
    loaded_txt.shape.should eq([3, 2])
    loaded_txt.to_unsafe[5].should eq(6.0)

    # Cleanup
    File.delete(filename_bin)
    File.delete(filename_txt)
  end

  # 7. Ufunc Reduction Helpers
  it "accumulates elements and performs ufunc outer computations" do
    # accumulate
    a = Tensor.new([4]) { |i| i.to_f64 + 1.0 } # [1, 2, 3, 4]
    acc = a.accumulate(:add)
    acc.to_a.should eq([1.0, 3.0, 6.0, 10.0])

    # ufunc_outer
    v1 = Tensor.new([2]) { |i| i.to_f64 + 1.0 } # [1, 2]
    v2 = Tensor.new([3]) { |i| i.to_f64 + 2.0 } # [2, 3, 4]
    u_out = Tensor.ufunc_outer(v1, v2, :add)
    u_out.shape.should eq([2, 3])
    u_out.to_unsafe[0].should eq(3.0)
    u_out.to_unsafe[5].should eq(6.0)
  end

  # 8. Masked Tensors
  it "performs operations on masked tensors ignoring masked values" do
    data = Tensor.new([4]) { |i| i.to_f64 + 1.0 } # [1, 2, 3, 4]
    mask = Tensor.from_array([false, true, false, true]) # mask elements 2 and 4 (indices 1 and 3)

    mt = Num::MaskedTensor.new(data, mask)
    mt.sum.should eq(4.0) # 1 + 3 = 4
    mt.mean.should eq(2.0) # (1 + 3) / 2 = 2.0
    mt.to_a.should eq([1.0, 3.0])

    # addition combines masks
    mt2 = Num::MaskedTensor.new(data, mask)
    added = mt + mt2
    added.sum.should eq(8.0)
    added.mask.to_a.should eq([false, true, false, true])
  end

  # 9. Sparse Matrices
  it "performs sparse matrix-vector multiplication" do
    rows = Tensor.from_array([0, 1, 2])
    cols = Tensor.from_array([0, 2, 1])
    data = Tensor.from_array([5.0, 10.0, 15.0])
    # matrix is:
    # [5, 0, 0]
    # [0, 0, 10]
    # [0, 15, 0]
    shape = [3, 3]

    coo = Num::SparseCOOTensor.new(rows, cols, data, shape)
    vec = Tensor.from_array([1.0, 2.0, 3.0])
    
    res = coo.matmul(vec)
    res.to_a.should eq([5.0, 30.0, 30.0])
  end
end
