# Polynomial mathematics module

module Num
  class Polynomial(T)
    getter coeffs : Array(T)

    def initialize(@coeffs : Array(T))
      while @coeffs.size > 1 && @coeffs.last == T.zero
        @coeffs.pop
      end
    end

    def degree : Int32
      @coeffs.size - 1
    end

    # Evaluate polynomial at a scalar value x using Horner's method
    def eval(x : T) : T
      result = @coeffs.last
      (degree - 1).step(to: 0, by: -1) do |i|
        result = result * x + @coeffs[i]
      end
      result
    end

    # Evaluate polynomial at a Tensor x using Horner's method
    def eval(x : Tensor(T, S)) : Tensor(T, S) forall S
      result = Tensor(T, S).new(x.shape, @coeffs.last)
      (degree - 1).step(to: 0, by: -1) do |i|
        result = result * x + @coeffs[i]
      end
      result
    end

    # Add two polynomials
    def +(other : Polynomial(T)) : Polynomial(T)
      new_size = {self.coeffs.size, other.coeffs.size}.max
      new_coeffs = Array(T).new(new_size) do |i|
        v1 = i < self.coeffs.size ? self.coeffs[i] : T.zero
        v2 = i < other.coeffs.size ? other.coeffs[i] : T.zero
        v1 + v2
      end
      Polynomial(T).new(new_coeffs)
    end

    # Multiply two polynomials
    def *(other : Polynomial(T)) : Polynomial(T)
      new_size = self.degree + other.degree + 1
      new_coeffs = Array(T).new(new_size, T.zero)
      self.coeffs.each_with_index do |c1, i|
        other.coeffs.each_with_index do |c2, j|
          new_coeffs[i + j] += c1 * c2
        end
      end
      Polynomial(T).new(new_coeffs)
    end

    # Compute derivative of polynomial
    def derivative : Polynomial(T)
      if degree == 0
        return Polynomial(T).new([T.zero])
      end
      new_coeffs = Array(T).new(degree) do |i|
        @coeffs[i + 1] * (i + 1)
      end
      Polynomial(T).new(new_coeffs)
    end
  end
end
