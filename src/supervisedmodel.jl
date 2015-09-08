export SupervisedModel
export iterations, cost, fitness, predict, predictProb, accuracy
export stop!, state, remember!, history, trainingCurve, name
export @remember!, @history

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
  _state::State
  _iterations::Int
  _native::T
  _history::TrainingHistory{Int}
end

function SupervisedModel{T<:Any}(model::T)
  SupervisedModel{T}(:uninitialized, zero(Int), model, TrainingHistory(Int))
end

# ==========================================================================
# Macros for simpler function defintion

macro _static_always(f, command, args...)
  esc(:(($f{T<:Any})(model::SupervisedModel{T},$(args...)) = $command))
end

macro _static_training_only(f, command, args...)
  msg = string(f, " only defined during model training")
  esc(:(($f{T<:Any})(model::SupervisedModel{T},$(args...)) = :training <= model._state <= :trystop ? $command : throw(StateError($msg))))
end

macro _static_initialized_only(f, command, args...)
  msg = string(f, " not defined for uninitialized or erroneous models")
  esc(:(($f{T<:Any})(model::SupervisedModel{T},$(args...)) = model._state < :initialized ? throw(StateError($msg)) : $command))
end

macro _delegate_initialized_only(f, args...)
  vars = map(x->x.args[1], args)
  msg = string(f, " not defined for uninitialized or erroneous models")
  esc(:(($f{T<:Any})(model::SupervisedModel{T},$(args...)) = model._state < :initialized ? throw(StateError($msg)) : $f(model._native,$(vars...))))
end

# ==========================================================================
# Define the local and delegating functions

@_static_always(name, string(typeof(model._native)))
@_static_always(state, model._state) # :untrained, :training, :trained
@_static_always(iterations, model._iterations)
@_static_training_only(stop!, model._state = :trystop)
@_static_training_only(remember!, push!(model._history, model._iterations, key, value), key::Symbol, value)
@_static_initialized_only(history, get(model._history, key), key::Symbol)
@_static_initialized_only(trainingCurve, get(model._history, :trainingCurve))
@_delegate_initialized_only(cost)
@_delegate_initialized_only(cost, data::DataSource)
@_delegate_initialized_only(fitness)
@_delegate_initialized_only(fitness, data::DataSource)
@_delegate_initialized_only(predict, data::DataSource)
@_delegate_initialized_only(predictProb, data::DataSource)
@_delegate_initialized_only(accuracy, data::DataSource)

macro remember!(model, func)
  key = string(func)
  esc(:(@remember!($model, symbol($key), $func)))
end

macro history(model, func)
  key = string(func)
  esc(:(@get($model, symbol($key))))
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
    break_every::Int = -1, # -1 = automatic, 0 = off
    nargs...)
  @assert max_iter > 0
  break_every = break_every < 0 ? safeFloor(max_iter / 10) : break_every
  enable_callback = break_every > 0

  # Define callback for the native model
  function localCallback()
    model._iterations += 1
    if model._iterations % break_every == 0
      remember!(model, :trainingCurve, cost(model)) # This should always be trivial cheap
      userCallback()
      model._state._idx == @stateidx(:trystop) ? :stop : nothing
    end
  end

  # If callback is disabled only pass the empty function
  cb = enable_callback ? localCallback : defaultCallback

  # Train the native model
  model._state = :training
  try 
    itersOld = model._iterations
    itersNew = train!(cb, model._native, data, solver, args...; max_iter=max_iter, nargs...)
    model._iterations = itersOld + itersNew
    model._state = :trained
  catch ex
    print("An exception was thrown while training: ", ex)
    model._state = :error
  end
  model
end
