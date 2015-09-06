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
  LogisticRegression(l1_coef, l2_coef, nothing)
end

end

typealias LogReg Classifier.LogisticRegression


# ==========================================================================
# Implement the interface

using SupervisedLearning.Classifier

iterations(model::EmpiricalRiskLearner) = model._iterations
state(model::EmpiricalRiskLearner) = model.state
cost(model::EmpiricalRiskLearner) = model._cost

function cost(model::EmpiricalRiskLearner, data::EncodedInMemoryLabeledDataSource)
  X = features(data)
  t = targets(data)
  rmodel = model._native_model.rmodel
  f = if typeof(model.regularizer) == EmpiricalRisks.ZeroReg
    Regression.RiskFun(rmodel, X, t)
  else
    Regression.RegRiskFun(rmodel, model.regularizer, X, t)
  end
  mean(value(f, model._coefficients))
end

function predict(model::EmpiricalRiskLearner{LogisticLoss}, data::EncodedInMemoryLabeledDataSource)
  pm = model._native_model.rmodel.predmodel
  pred = sign(EmpiricalRisks.predict(pm, model._coefficients, features(data)))
  labeldecode(data.encoding, pred)
end

function predictProb(model::EmpiricalRiskLearner{LogisticLoss}, data::EncodedInMemoryLabeledDataSource)
  pm = model._native_model.rmodel.predmodel
  pred = tanh(EmpiricalRisks.predict(pm, model._coefficients, features(data)))
  pred / 2 + .5
end

function accuracy(model::EmpiricalRiskLearner, data::DataSource)
  gt = groundtruth(data)
  correct = sum(predict(model, data) .== gt)
  correct / length(gt)
end

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

iterations(model::LogReg) =
  model._risk_learner == nothing ? 0 : iterations(model._risk_learner)

state(model::LogReg) =
  model._risk_learner == nothing ? :untrained : state(model._risk_learner)

cost(model::LogReg) =
  model._risk_learner == nothing ? typemax(Float64) : cost(model._risk_learner)

cost(model::LogReg, data::DataSource) =
  model._risk_learner == nothing ? typemax(Float64) : cost(model._risk_learner, data)

predict(model::LogReg, data::DataSource) =
  model._risk_learner == nothing ? nothing : predict(model._risk_learner, data)

predictProb(model::LogReg, data::DataSource) =
  model._risk_learner == nothing ? nothing : predictProb(model._risk_learner, data)

accuracy(model::Any, data::DataSource) =
  model._risk_learner == nothing ? nothing : accuracy(model._risk_learner, data)

trainingCurve(model::LogReg) =
  model._risk_learner == nothing ? throw(ArgumentError("Cannot call history on untrained model")) : trainingCurve(model._risk_learner)

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

function train!{E<:SignedClassEncoding, L<:UnivariateLoss, G<:Any}(
    callback::Function,
    model::EmpiricalRiskLearner{L},
    data::EncodedInMemoryLabeledDataSource{E,G,1},
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
  f = if typeof(model.regularizer) == EmpiricalRisks.ZeroReg
    Regression.RiskFun(rmodel, X, t)
  else
    Regression.RegRiskFun(rmodel, model.regularizer, X, t)
  end

  # inital theta
  theta = typeof(rmodel.predmodel) == LinearPred ? zeros(n) : zeros(n+1)

  # Reset training _history
  model.state = :training
  model._native_model = f
  model._history = Dict{String,Dict{Int,Any}}()
  model._history["cost"] = Dict{Int,Float64}()
  model._history["coefficients"] = Dict{Int,Array}()
  model._iterations = 0
  model._coefficients = theta
  setModelInternals!(model, data, theta, 1, cost(model, data))

  # define callback
  cb = logRegCallback(model, data, callback, break_every, tell_user = true)

  # Fit model
  r = Regression.solve!(Regression.GD(), f, theta,
                        Regression.Options(maxiter = max_iter),
                        cb)

  model.state = :trained
  setModelInternals!(model, data, r.sol, r.niters, r.fval)
  model
end

function train!{E<:ClassEncoding, G<:Any}(
    callback::Function,
    model::LogisticRegression,
    data::EncodedInMemoryLabeledDataSource{E,G,1},
    solver::Solver.GradientDescent;
    args...)
  if model.l1_coef == model.l2_coef == 0.
    # No reg
    newModel = EmpiricalRiskLearner(LogisticLoss(), EmpiricalRisks.ZeroReg())
    model._risk_learner = newModel
    train!(callback, newModel, data, solver; args...)
  elseif model.l1_coef == 0. && model.l2_coef != 0.
    # Only l2 reg
    newModel = EmpiricalRiskLearner(LogisticLoss(), EmpiricalRisks.SqrL2Reg(model.l2_coef))
    model._risk_learner = newModel
    train!(callback, newModel, data, solver; args...)
  elseif model.l1_coef != 0. && model.l2_coef == 0.
    # Only l1 reg
    newModel = EmpiricalRiskLearner(LogisticLoss(), EmpiricalRisks.L1Reg(model.l1_coef))
    model._risk_learner = newModel
    train!(callback, newModel, data, solver; args...)
  else
    throw(ArgumentError("Hyperparameter of LogisticRegression have an illegal combination"))
  end
end
