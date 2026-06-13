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
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
# LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
# WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

require "../spec_helper"

describe Tensor do
  {% if flag?(:arrow) %}
    it "performs in-place addition on ARROW backend using compute engine" do
      a = Tensor(Int32, ARROW(Int32)).new([3], device: ARROW(Int32)) { |i| i }
      b = Tensor(Int32, ARROW(Int32)).new([3], device: ARROW(Int32)) { |i| 10 }
      
      Num.add!(a, b)
      a.to_a.should eq([10, 11, 12])
    end

    it "performs in-place subtraction on ARROW backend using compute engine" do
      a = Tensor(Int32, ARROW(Int32)).new([3], device: ARROW(Int32)) { |i| i + 10 }
      b = Tensor(Int32, ARROW(Int32)).new([3], device: ARROW(Int32)) { |i| 2 }
      
      Num.subtract!(a, b)
      a.to_a.should eq([8, 9, 10])
    end

    it "performs in-place multiplication on ARROW backend using compute engine" do
      a = Tensor(Int32, ARROW(Int32)).new([3], device: ARROW(Int32)) { |i| i + 1 }
      b = Tensor(Int32, ARROW(Int32)).new([3], device: ARROW(Int32)) { |i| 5 }
      
      Num.multiply!(a, b)
      a.to_a.should eq([5, 10, 15])
    end

    it "performs in-place division on ARROW backend using compute engine" do
      a = Tensor(Float64, ARROW(Float64)).new([3], device: ARROW(Float64)) { |i| (i + 1) * 10.0 }
      b = Tensor(Float64, ARROW(Float64)).new([3], device: ARROW(Float64)) { |i| 2.0 }
      
      Num.divide!(a, b)
      a.to_a.should eq([5.0, 10.0, 15.0])
    end

    it "performs unary negation on ARROW backend using compute engine" do
      a = Tensor(Int32, ARROW(Int32)).new([3], device: ARROW(Int32)) { |i| i + 1 }
      res = -a
      res.to_a.should eq([-1, -2, -3])
    end

    it "converts a CPU tensor to ARROW using .arrow" do
      a = [1, 2, 3].to_tensor
      a_arrow = a.arrow
      a_arrow.is_a?(Tensor(Int32, ARROW(Int32))).should be_true
      a_arrow.to_a.should eq([1, 2, 3])
    end
  {% end %}
end
