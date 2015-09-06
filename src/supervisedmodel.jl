export iterations, state, trainingCurve, cost, fitness, predict, predictProb, accuracy

function defaultCallback()
end

# ==========================================================================
# Interface for all models

iterations(model::Any) = @_not_implemented
state(model::Any) = @_not_implemented
trainingCurve(model::Any) = @_not_implemented
cost(model::Any) = @_not_implemented
fitness(model::Any) = -1 * cost(model)
predict(model::Any, data::DataSource) = @_not_implemented
predictProb(model::Any, data::DataSource) = @_not_implemented
accuracy(model::Any, data::DataSource) = @_not_implemented
train!(callback::Function, model::Any, data::DataSource, solver::AbstractSolver, args...; nargs...) = @_not_implemented


# ==========================================================================
# Use default callback if no other is specified

train!(model::Any, data::DataSource, solver::AbstractSolver, args...; nargs...) =
  train!(defaultCallback, model, data, solver, args...; nargs...)
