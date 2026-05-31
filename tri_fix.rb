content = File.read("src/linalg/linalg.cr")

triu_new = <<-CRYSTAL
  def triu!(k : Int = 0)
    m, n = @shape
    ptr = self.to_unsafe
    m.times do |i|
      n.times do |j|
        if i > j - k
          ptr[i * n + j] = T.zero
        end
      end
    end
    self
  end
CRYSTAL

tril_new = <<-CRYSTAL
  def tril!(k : Int = 0)
    m, n = @shape
    ptr = self.to_unsafe
    m.times do |i|
      n.times do |j|
        if i < j - k
          ptr[i * n + j] = T.zero
        end
      end
    end
    self
  end
CRYSTAL

# Surgical replace for triu!
content.gsub!(/def triu!\(k : Int = 0\).*?end/m, triu_new)
# Surgical replace for tril!
content.gsub!(/def tril!\(k : Int = 0\).*?end/m, tril_new)

File.write("src/linalg/linalg.cr", content)
