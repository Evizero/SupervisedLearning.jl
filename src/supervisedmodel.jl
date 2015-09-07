export SupervisedModel
export iterations, cost, fitness, predict, predictProb, accuracy
export state, remember!, @remember!, history, trainingCurve, name

import StatsBase.predict

function defaultCallback()
end

# ==========================================================================
# Interface for all models

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
  _state::Symbol
  _iterations::Int
  _native::T
  _history::TrainingHistory{Int}
end

function SupervisedModel{T<:Any}(model::T)
  SupervisedModel{T}(:untrained, zero(Int), model, TrainingHistory(Int))
end

# ==========================================================================
# Macros for simpler function defintion

macro _static_always(f, command, args...)
  esc(:(($f{T<:Any})(model::SupervisedModel{T},$(args...)) = $command))
end

macro _static_training_only(f, command, args...)
  if length(args) > 0 && args[end].args[1] == :nargs
    esc(:(($f{T<:Any})(model::SupervisedModel{T},$(args[1:end-1]...);nargs...) = model._state == :training ? $command : throw(ArgumentError("Function is only defined while the model is in training"))))
  else
    esc(:(($f{T<:Any})(model::SupervisedModel{T},$(args...)) = model._state == :training ? $command : throw(ArgumentError("Function is only defined while the model is in training"))))
  end
end

macro _static_not_untrained_only(f, command, args...)
  esc(:(($f{T<:Any})(model::SupervisedModel{T},$(args...)) = model._state == :untrained ? throw(ArgumentError("Function not defined for untrained models")) : $command))
end

macro _delegate_not_untrained_only(f, args...)
  local vars = map(x->x.args[1], args)
  esc(:(($f{T<:Any})(model::SupervisedModel{T},$(args...)) = model._state == :untrained ? throw(ArgumentError("Function not defined for untrained models")) : $f(model._native,$(vars...))))
end

macro _delegate_not_untrained_else(f, alternative, args...)
  local vars = map(x->x.args[1], args)
  esc(:(($f{T<:Any})(model::SupervisedModel{T},$(args...)) = model._state == :untrained ? $alternative : $f(model._native,$(vars...))))
end

# ==========================================================================
# Define the local and delegating functions

@_static_always(name, string(typeof(model._native)))
@_static_always(state, model._state) # :untrained, :training, :trained
@_static_always(iterations, model._iterations)
@_static_training_only(remember!, push!(model._history, model._iterations, f, args...; nargs...), f::Function, args..., nargs...)
@_static_not_untrained_only(history, get(model._history, f), f::Function)
@_static_not_untrained_only(trainingCurve, get(model._history, cost))
@_delegate_not_untrained_only(cost)
@_delegate_not_untrained_only(cost, data::DataSource)
@_delegate_not_untrained_only(fitness)
@_delegate_not_untrained_only(fitness, data::DataSource)
@_delegate_not_untrained_only(predict, data::DataSource)
@_delegate_not_untrained_only(predictProb, data::DataSource)
@_delegate_not_untrained_only(accuracy, data::DataSource)

macro remember!(model, func)
  esc(:(remember!($model, $(func.args...))))
end

# ==========================================================================
# Train method

function train!{T<:Any, D<:LabeledDataSource}(
    userCallback::Function,
    model::SupervisedModel{T},
    data::D,
    solver::AbstractSolver,
    args...;
    max_iter::Int = 1000,
    break_every::Int = -1, # -1 ... automatic, 0 ... off
    nargs...)
  @assert max_iter > 0
  break_every = break_every < 0 ? safeFloor(max_iter / 10) : break_every
  enable_callback = break_every > 0

  # Define callback for the native model
  function localCallback()
    model._iterations += 1
    if model._iterations % break_every == 0
      @remember!(model, cost(model)) # This should always be cheap
      userCallback()
    end
  end

  # If callback is disabled only pass the empty function
  cb = enable_callback ? localCallback : defaultCallback

  # Train native model
  model._state = :training
  itersOld = model._iterations
  itersNew = train!(cb, model._native, data, solver, args...; max_iter=max_iter, nargs...)
  model._iterations = itersOld + itersNew
  model._state = :trained
  model
end
