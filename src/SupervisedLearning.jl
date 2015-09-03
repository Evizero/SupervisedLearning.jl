module SupervisedLearning

using Reexport
@reexport using MLBase

include("common.jl")
include("classencoding.jl")
include("datasource.jl")

end # module
