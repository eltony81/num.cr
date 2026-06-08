# Tensor contractions and vector products

class Tensor(T, S)
  # Computes the outer product of two tensors (inputs are flattened if not 1D)
  def self.outer(a : Tensor(T, S), b : Tensor(T, S)) : Tensor(T, S)
    a_flat = a.reshape([a.size])
    b_flat = b.reshape([b.size])
    
    result = Tensor(T, S).zeros([a_flat.size, b_flat.size])
    a_flat.size.times do |i|
      b_flat.size.times do |j|
        result.to_unsafe[i * b_flat.size + j] = a_flat.to_unsafe[i] * b_flat.to_unsafe[j]
      end
    end
    result
  end

  # Computes the cross product of two 3-element vectors
  def self.cross(a : Tensor(T, S), b : Tensor(T, S)) : Tensor(T, S)
    if a.size != 3 || b.size != 3
      raise Num::Exceptions::ValueError.new("Cross product is only supported for 3-element vectors")
    end
    
    result = Tensor(T, S).zeros([3])
    result.to_unsafe[0] = a.to_unsafe[1] * b.to_unsafe[2] - a.to_unsafe[2] * b.to_unsafe[1]
    result.to_unsafe[1] = a.to_unsafe[2] * b.to_unsafe[0] - a.to_unsafe[0] * b.to_unsafe[2]
    result.to_unsafe[2] = a.to_unsafe[0] * b.to_unsafe[1] - a.to_unsafe[1] * b.to_unsafe[0]
    result
  end
end
