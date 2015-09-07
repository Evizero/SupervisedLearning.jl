using SupervisedLearning
using Base.Test

macro spc_time(expr)
  quote
    $expr # compile
    print("       - ")
    @time $expr
    println()
  end
end

function msg(args...)
  println("   --> ", args...)
end

tests = [
  "tst_classencoding.jl"
  "tst_datasource.jl"
  "tst_traininghistory.jl"
  "tst_integration.jl"
  "tst_empiricalrisks.jl"
]

perf = [
  #"bm_datasource.jl"
]

for t in tests
  println("[->] $t")
  include(t)
  println("[OK] $t")
  println("====================================================================")
end

for p in perf
  println("[->] $p")
  include(p)
  println("[OK] $p")
  println("====================================================================")
end
