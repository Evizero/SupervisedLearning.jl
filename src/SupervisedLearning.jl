module SupervisedLearning

function safeRound(num)
  if VERSION < v"0.4-"
    iround(num)
  else
    round(Integer,num)
  end
end

include("classencoding.jl")
include("datasource.jl")

end # module
