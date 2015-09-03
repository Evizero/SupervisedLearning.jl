module SupervisedLearning

using Reexport
@reexport using MLBase

function safeRound(num)
  if VERSION < v"0.4-"
    iround(num)
  else
    round(Integer, num)
  end
end

function safeFloor(num)
  if VERSION < v"0.4-"
    ifloor(num)
  else
    floor(Integer, num)
  end
end

include("classencoding.jl")
include("datasource.jl")

end # module
