using SupervisedLearning
using Base.Test

function msg(args...)
  println("   --> ", args...)
end

tests = [
  "tst_datasource.jl"
]

for t in tests
  println("[->] $t")
  include(t)
  println("[OK] $t")
  println("====================================================================")
end
