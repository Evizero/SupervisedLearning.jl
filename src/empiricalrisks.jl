export Classifier
export LogReg
export trainingCurve

using EmpiricalRisks
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
                          :untrained, [.0], .0, 0, nothing, Dict())
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

using SupervisedLearning.Classifier

function trainingCurve(model::EmpiricalRiskLearner)
  if(!haskey(model._history, "cost"))
    throw(ArgumentError("Cannot call history on untrained model"))
  end
  costDict = model._history["cost"]
  iterations = sort([k for k in keys(costDict)])
  costValues = convert(Vector{Float64}, [costDict[i] for i in iterations])
  (iterations, costValues)
end

function trainingCurve(model::LogReg)
  trainingCurve(model._risk_learner)
end

# ==========================================================================

function setModelInternals!{L<:Loss}(model::EmpiricalRiskLearner{L}, data::InMemoryLabeledDataSource, theta, iter, cost)
  model._coefficients = theta
  model._history["coefficients"][iter] = theta
  tCost = mean(value(model._native_model, theta, features(data), targets(data)))
  model._cost = tCost
  model._history["cost"][iter] = tCost
  model._iterations = iter
end

function setModelInternals!{L<:Loss}(model::EmpiricalRiskLearner{L}, theta)
  model._coefficients = theta
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
    max_iter = 1000,
    break_every = 10,
    args...)
  X = features(data)
  t = targets(data)
  b = bias(data)
  m = nobs(data)
  n = nvar(data)
  rmodel = if(b == 0.0)
    riskmodel(LinearPred(n), model.lossfunction)
  else
    riskmodel(AffinePred(n, b), model.lossfunction)
  end
  f = Regression.RegRiskFun(rmodel, model.regularizer, X, t)

  # Reset training _history
  model.state = :training
  model._native_model = rmodel
  model._history = Dict{String,Dict{Int,Any}}()
  model._history["cost"] = Dict{Int,Float64}()
  model._history["coefficients"] = Dict{Int,Array}()
  model._iterations = 0

  # define callback
  cb = logRegCallback(model, data, callback, break_every, tell_user = true)

  # inital theta
  theta = typeof(rmodel.predmodel) == LinearPred ? randn(n) : randn(n+1)

  # Fit model
  Regression.solve!(GD(), f, theta,
                    Regression.Options(maxiter = max_iter),
                    cb)

  model.state = :trained
  setModelInternals!(model, theta)
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
