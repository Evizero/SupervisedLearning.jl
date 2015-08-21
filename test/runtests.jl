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
  "tst_datasource.jl"
  "bm_datasource.jl"
]

for t in tests
  println("[->] $t")
  include(t)
  println("[OK] $t")
  println("====================================================================")
end
