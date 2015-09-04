export AbstractSolver, Solver
export train!

abstract AbstractSolver

# ==========================================================================
# Namespacing for convenient enumerations
module Solver

using ..SupervisedLearning.AbstractSolver

immutable GradientDescent <: AbstractSolver
end

immutable BFGS <: AbstractSolver
end

end
