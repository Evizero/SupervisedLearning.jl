module SupervisedLearning

export train!

using Reexport
@reexport using MLBase

include("common.jl")
include("classencoding.jl")
include("datasource.jl")
include("solver.jl")
include("empiricalrisks.jl")

end # module
