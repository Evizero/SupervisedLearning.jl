import StatsBase.predict

export iterations, state, trainingCurve, cost, fitness, predict, predictProb, accuracy

function defaultCallback()
end

# ==========================================================================
# Interface for all models

iterations(model::Any) = @_not_implemented
state(model::Any) = @_not_implemented # :untrained, :training, :trained
trainingCurve(model::Any) = @_not_implemented
cost(model::Any) = @_not_implemented
cost(model::Any, data::DataSource) = @_not_implemented
fitness(model::Any) = -1 * cost(model)
fitness(model::Any, data::DataSource) = -1 * cost(model, data)
#predict(model::Any, data::DataSource) = @_not_implemented # Statsbase takes care of this
predictProb(model::Any, data::DataSource) = @_not_implemented
accuracy(model::Any, data::DataSource) = @_not_implemented
train!(callback::Function, model::Any, data::DataSource, solver::AbstractSolver, args...; nargs...) = @_not_implemented

# ==========================================================================
# Use default callback if no other is specified

train!(model::Any, data::DataSource, solver::AbstractSolver, args...; nargs...) =
  train!(defaultCallback, model, data, solver, args...; nargs...)

# ==========================================================================
# Embedding supervised learning model for convenience

type SupervisedModel{T<:Any}
  _native::T
  _state::Symbol
  _history::TrainingHistory{Int}
end

# ==========================================================================
# Macros for simpler function defintion

macro _static_always(f, command, args...)
  local vars = map(x->x.args[1], args)
  esc(:(($f{T<:Any})(model::SupervisedModel{T},$(args...)) = $command))
end

macro _delegate_trained_only(f, args...)
  local vars = map(x->x.args[1], args)
  esc(:(($f{T<:Any})(model::SupervisedModel{T},$(args...)) = model._state == :untrained ? throw(ArgumentError("Function not defined for untrained models")) : $f(model._native,$(vars...))))
end

macro _delegate_trained_else(f, alternative, args...)
  local vars = map(x->x.args[1], args)
  esc(:(($f{T<:Any})(model::SupervisedModel{T},$(args...)) = model._state == :untrained ? $alternative : $f(model._native,$(vars...))))
end

# ==========================================================================
# Define the local and delegating functions

@_static_always(state, model._state)

@_delegate_trained_else(iterations, 0)

@_delegate_trained_only(cost)
@_delegate_trained_only(cost, data::DataSource)
@_delegate_trained_only(fitness)
@_delegate_trained_only(fitness, data::DataSource)
@_delegate_trained_only(predict, data::DataSource)
@_delegate_trained_only(predictProb, data::DataSource)
@_delegate_trained_only(accuracy, data::DataSource)
