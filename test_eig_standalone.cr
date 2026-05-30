require "./src/num"
require "complex"

# We re-implement the dgeev call here to debug
lib LibLapack
  fun dgeev = dgeev_(jobvl : UInt8*, jobvr : UInt8*, n : Int32*, a : Float64*, lda : Int32*, wr : Float64*, wi : Float64*, vl : Float64*, ldvl : Int32*, vr : Float64*, ldvr : Int32*, work : Float64*, lwork : Int32*, info : Int32*) : Int32
end

def debug_eig(m)
  a = m.dup(Num::ColMajor)
  n = a.shape[0].to_i32
  wr = Pointer(Float64).malloc(n)
  wi = Pointer(Float64).malloc(n)
  
  jobvl = 'N'.ord.to_u8
  jobvr = 'N'.ord.to_u8
  lda = n
  ldvl = 1
  ldvr = 1
  vl_dummy = 0.0
  vr_dummy = 0.0
  
  # Workspace query
  lwork = -1
  work_query = 0.0
  info = 0
  LibLapack.dgeev(pointerof(jobvl), pointerof(jobvr), pointerof(n), a.to_unsafe, pointerof(lda),
    wr, wi, pointerof(vl_dummy), pointerof(ldvl), pointerof(vr_dummy), pointerof(ldvr), 
    pointerof(work_query), pointerof(lwork), pointerof(info))
  
  lwork = work_query.to_i32
  work = Pointer(Float64).malloc(lwork)
  
  LibLapack.dgeev(pointerof(jobvl), pointerof(jobvr), pointerof(n), a.to_unsafe, pointerof(lda),
    wr, wi, pointerof(vl_dummy), pointerof(ldvl), pointerof(vr_dummy), pointerof(ldvr), 
    work, pointerof(lwork), pointerof(info))
    
  puts "INFO: #{info}"
  n.times do |i|
    puts "Root #{i}: #{wr[i]} + #{wi[i]}j"
  end
end

# Test case from cryspace failure
m = [[-3.0, -2.0], [1.0, 0.0]].to_tensor
puts "Testing companion matrix:"
debug_eig(m)

m2 = [[0.0, 1.0], [-10.0, -0.5]].to_tensor
puts "\nTesting oscillator matrix:"
debug_eig(m2)
