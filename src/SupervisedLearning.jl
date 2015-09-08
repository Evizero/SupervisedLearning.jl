module SupervisedLearning

export train!

using Reexport
@reexport using MLBase

include("common.jl")
include("state.jl")
include("classencoding.jl")
include("datasource.jl")
include("traininghistory.jl")
include("solver.jl")
include("supervisedmodel.jl")
include("empiricalrisks.jl")

end # module
