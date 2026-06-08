# Ufunc Reduction and Operational Helpers

class Tensor(T, S)
  # Performs a running/accumulated operation across flat elements of the tensor
  def accumulate(op : Symbol) : Tensor(T, S)
    result = Tensor(T, S).zeros(self.shape)
    if self.size == 0
      return result
    end
    
    current = self.to_unsafe[0]
    result.to_unsafe[0] = current
    
    (1...self.size).each do |i|
      val = self.to_unsafe[i]
      case op
      when :add
        current = current + val
      when :multiply
        current = current * val
      when :min
        current = val < current ? val : current
      when :max
        current = val > current ? val : current
      else
        raise Num::Exceptions::ValueError.new("Unknown accumulate operator: #{op}")
      end
      result.to_unsafe[i] = current
    end
    result
  end

  # Performs an outer element-wise operation between two tensors
  def self.ufunc_outer(a : Tensor(T, S), b : Tensor(T, S), op : Symbol) : Tensor(T, S)
    a_flat = a.reshape([a.size])
    b_flat = b.reshape([b.size])
    
    result = Tensor(T, S).zeros([a_flat.size, b_flat.size])
    a_flat.size.times do |i|
      b_flat.size.times do |j|
        v1 = a_flat.to_unsafe[i]
        v2 = b_flat.to_unsafe[j]
        val = case op
              when :add      then v1 + v2
              when :subtract then v1 - v2
              when :multiply then v1 * v2
              when :divide   then v1 / v2
              else
                raise Num::Exceptions::ValueError.new("Unknown ufunc_outer operator: #{op}")
              end
        result.to_unsafe[i * b_flat.size + j] = val
      end
    end
    result
  end
end
