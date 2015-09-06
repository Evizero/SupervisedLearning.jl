export Classifier
export LogReg

import EmpiricalRisks
using Regression

# ==========================================================================
# Namespacing for convenient enumerations
module Classifier

export EmpiricalRiskLearner, LogisticRegression

using EmpiricalRisks
using Regression

type EmpiricalRiskLearner{L<:Loss}
  lossfunction::L
  regularizer::Regularizer
  state::Symbol
  _coefficients::Vector{Float64}
  _cost::Float64
  _iterations::Int
  _native_model::Any
  _history::Dict{String,Dict{Int,Any}}
end

function EmpiricalRiskLearner{L<:Loss}(lossfunction::L, regularizer::Regularizer)
  EmpiricalRiskLearner{L}(lossfunction, regularizer,
                          :untrained, [.0], typemax(Float64), 0, nothing, Dict())
end

# --------------------------------------------------------------------------

type LogisticRegression
  l1_coef::Float64
  l2_coef::Float64
  _risk_learner::Any
end

function LogisticRegression(; l1_coef = .0, l2_coef = .0)
  # todo select regu based on params
  LogisticRegression(l1_coef, l2_coef, nothing)
  #LogisticRegression(LogisticLoss(), EmpiricalRisks.SqrL2Reg(l2_coef))
end

end

typealias LogReg Classifier.LogisticRegression


# ==========================================================================
# Implement the interface

using SupervisedLearning.Classifier

iterations(model::EmpiricalRiskLearner) = model._iterations
state(model::EmpiricalRiskLearner) = model.state
cost(model::EmpiricalRiskLearner) = model._cost

function trainingCurve(model::EmpiricalRiskLearner)
  if(!haskey(model._history, "cost"))
    throw(ArgumentError("Cannot call history on untrained model"))
  end
  costDict = model._history["cost"]
  iterations = sort([k for k in keys(costDict)])
  costValues = convert(Vector{Float64}, [costDict[i] for i in iterations])
  (iterations, costValues)
end

# --------------------------------------------------------------------------

iterations(model::LogReg) = model._risk_learner == nothing ? 0 : iterations(model._risk_learner)
state(model::LogReg) = model._risk_learner == nothing ? :untrained : state(model._risk_learner)
cost(model::LogReg) = model._risk_learner == nothing ? typemax(Float64) : cost(model._risk_learner)
trainingCurve(model::LogReg) = trainingCurve(model._risk_learner)

# ==========================================================================
# Implement the train! functions

function setModelInternals!{L<:Loss}(model::EmpiricalRiskLearner{L}, data::InMemoryLabeledDataSource, theta, iter, cost)
  model._coefficients = theta
  model._history["coefficients"][iter] = theta
  #tCost = mean(value(model._native_model, theta))
  model._cost = cost
  model._history["cost"][iter] = cost
  model._iterations = iter
end

function logRegCallback{L<:Loss}(model::EmpiricalRiskLearner{L}, data::InMemoryLabeledDataSource, userCallback, breakEvery; tell_user = true)
  function callback(iter, theta, cost, grad)
    if iter % breakEvery == 0
      setModelInternals!(model, data, theta, iter, cost)
      if(tell_user)
        userCallback()
      end
    end
  end
end

function train!{E<:SignedClassEncoding, L<:UnivariateLoss}(
    callback::Function,
    model::EmpiricalRiskLearner{L},
    data::EncodedInMemoryLabeledDataSource{E,1},
    solver::Solver.GradientDescent;
    max_iter::Int = 1000,
    break_every::Int = -1,
    args...)
  @assert max_iter > 0
  break_every = break_every > 0 ? break_every : safeFloor(max_iter / 10)
  X = features(data)
  t = targets(data)
  b = bias(data)
  m = nobs(data)
  n = nvar(data)
  rmodel = if b == 0.0
    riskmodel(LinearPred(n), model.lossfunction)
  else
    riskmodel(AffinePred(n, b), model.lossfunction)
  end
  f = Regression.RegRiskFun(rmodel, model.regularizer, X, t)

  # Reset training _history
  model.state = :training
  model._native_model = f
  model._history = Dict{String,Dict{Int,Any}}()
  model._history["cost"] = Dict{Int,Float64}()
  model._history["coefficients"] = Dict{Int,Array}()
  model._iterations = 0

  # define callback
  cb = logRegCallback(model, data, callback, break_every, tell_user = true)

  # inital theta
  theta = typeof(rmodel.predmodel) == LinearPred ? zeros(n) : zeros(n+1)

  # Fit model
  r = Regression.solve!(Regression.GD(), f, theta,
                        Regression.Options(maxiter = max_iter),
                        cb)

  model.state = :trained
  setModelInternals!(model, data, r.sol, r.niters, r.fval)
  model
end

function train!{E<:ClassEncoding}(
    callback::Function,
    model::LogisticRegression,
    data::EncodedInMemoryLabeledDataSource{E,1},
    solver::Solver.GradientDescent;
    args...)
  newModel = EmpiricalRiskLearner(LogisticLoss(), EmpiricalRisks.SqrL2Reg(model.l2_coef))
  model._risk_learner = newModel
  train!(callback, newModel, data, solver; args...)
end
