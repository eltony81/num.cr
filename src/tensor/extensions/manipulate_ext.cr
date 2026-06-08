# Array manipulation extensions for Tensor

class Tensor(T, S)
  # Concatenates an array of tensors along a specified axis
  def self.concatenate(tensors : Array(Tensor(T, S)), axis : Int32 = 0) : Tensor(T, S)
    if tensors.empty?
      raise Num::Exceptions::ValueError.new("Cannot concatenate an empty list of Tensors")
    end
    t0 = tensors[0]
    rank = t0.rank
    
    # Normalize axis
    ax = axis < 0 ? rank + axis : axis
    if ax < 0 || ax >= rank
      raise Num::Exceptions::ValueError.new("Axis #{axis} out of bounds for rank #{rank}")
    end

    # Check that all shapes match except along the specified axis
    tensors.each do |t|
      if t.rank != rank
        raise Num::Exceptions::ValueError.new("All Tensors must have the same rank to concatenate")
      end
      rank.times do |i|
        next if i == ax
        if t.shape[i] != t0.shape[i]
          raise Num::Exceptions::ValueError.new("Tensors must have matching shape at axis #{i}")
        end
      end
    end

    # Compute new shape
    total_size_along_axis = tensors.sum &.shape[ax]
    new_shape = t0.shape.dup
    new_shape[ax] = total_size_along_axis

    result = Tensor(T, S).zeros(new_shape)
    
    offset = 0
    tensors.each do |t|
      if t.rank == 0
        result.to_unsafe[0] = t.to_unsafe[0]
      else
        nd = Num::Internal::NDIndex.new(t.shape)
        nd.each do |coords|
          src_offset = t.offset
          t.rank.times do |i|
            src_offset += coords[i] * t.strides[i]
          end
          
          dest_offset = result.offset
          t.rank.times do |i|
            coord_val = coords[i]
            coord_val += offset if i == ax
            dest_offset += coord_val * result.strides[i]
          end
          
          result.data.to_hostptr[dest_offset] = t.data.to_hostptr[src_offset]
        end
      end
      offset += t.shape[ax]
    end
    result
  end

  # Stack tensors horizontally (along axis 1, or axis 0 if 1D)
  def self.hstack(tensors : Array(Tensor(T, S))) : Tensor(T, S)
    if tensors.empty?
      raise Num::Exceptions::ValueError.new("Cannot hstack an empty array of Tensors")
    end
    t0 = tensors[0]
    axis = t0.rank <= 1 ? 0 : 1
    self.concatenate(tensors, axis)
  end

  # Stack tensors vertically (along axis 0)
  def self.vstack(tensors : Array(Tensor(T, S))) : Tensor(T, S)
    if tensors.empty?
      raise Num::Exceptions::ValueError.new("Cannot vstack an empty array of Tensors")
    end
    self.concatenate(tensors, 0)
  end

  # Cyclically shift elements along a given axis
  def roll(shift : Int32, axis : Int32 = 0) : Tensor(T, S)
    rank = self.rank
    if rank == 0
      return self.dup
    end
    ax = axis < 0 ? rank + axis : axis
    if ax < 0 || ax >= rank
      raise Num::Exceptions::ValueError.new("Axis #{axis} out of bounds")
    end

    result = Tensor(T, S).zeros(self.shape)
    shift_val = shift % self.shape[ax]
    if shift_val < 0
      shift_val += self.shape[ax]
    end

    nd = Num::Internal::NDIndex.new(self.shape)
    nd.each do |coords|
      src_offset = self.offset
      rank.times do |i|
        src_offset += coords[i] * self.strides[i]
      end

      dest_offset = result.offset
      rank.times do |i|
        coord_val = coords[i]
        if i == ax
          coord_val = (coord_val + shift_val) % self.shape[ax]
        end
        dest_offset += coord_val * result.strides[i]
      end

      result.data.to_hostptr[dest_offset] = self.data.to_hostptr[src_offset]
    end
    result
  end

  # Reverse the order of elements along a given axis
  def flip(axis : Int32 = 0) : Tensor(T, S)
    rank = self.rank
    if rank == 0
      return self.dup
    end
    ax = axis < 0 ? rank + axis : axis
    if ax < 0 || ax >= rank
      raise Num::Exceptions::ValueError.new("Axis #{axis} out of bounds")
    end

    result = Tensor(T, S).zeros(self.shape)

    nd = Num::Internal::NDIndex.new(self.shape)
    nd.each do |coords|
      src_offset = self.offset
      rank.times do |i|
        src_offset += coords[i] * self.strides[i]
      end

      dest_offset = result.offset
      rank.times do |i|
        coord_val = coords[i]
        if i == ax
          coord_val = self.shape[ax] - 1 - coord_val
        end
        dest_offset += coord_val * result.strides[i]
      end

      result.data.to_hostptr[dest_offset] = self.data.to_hostptr[src_offset]
    end
    result
  end
end
