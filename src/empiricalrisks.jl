export Classifier
export LogReg

import EmpiricalRisks
using Regression

# ==========================================================================
# Namespacing for convenient enumerations
module Classifier

export EmpiricalRiskLearner, LogisticRegression

using ..SupervisedLearning.SupervisedModel

using EmpiricalRisks
using Regression

type EmpiricalRiskLearner{L<:Loss}
  lossfunction::L
  regularizer::Regularizer
  _coefficients::Vector{Float64}
  _cost::Float64
  _native_model::Any
end

function EmpiricalRiskLearner{L<:Loss}(lossfunction::L, regularizer::Regularizer; box=true)
  model = EmpiricalRiskLearner{L}(lossfunction, regularizer, [.0], typemax(Float64), nothing)
  box ? SupervisedModel(model) : model
end

# --------------------------------------------------------------------------

type LogisticRegression
  l1_coef::Float64
  l2_coef::Float64
  _risk_learner::Any
end

function LogisticRegression(; l1_coef = .0, l2_coef = .0)
  SupervisedModel(LogisticRegression(l1_coef, l2_coef, nothing))
end

end

typealias LogReg Classifier.LogisticRegression


# ==========================================================================
# Implement the interface

using SupervisedLearning.Classifier

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

# --------------------------------------------------------------------------

cost(model::LogReg) = cost(model._risk_learner)
cost(model::LogReg, data::DataSource) = cost(model._risk_learner, data)
predict(model::LogReg, data::DataSource) = predict(model._risk_learner, data)
predictProb(model::LogReg, data::DataSource) = predictProb(model._risk_learner, data)
accuracy(model::LogReg, data::DataSource) = accuracy(model._risk_learner, data)

# ==========================================================================
# Implement the train! functions

function train!{E<:SignedClassEncoding, L<:UnivariateLoss, G<:Any}(
    callback::Function,
    model::EmpiricalRiskLearner{L},
    data::EncodedInMemoryLabeledDataSource{E,G,1},
    solver::Solver.GradientDescent,
    args...;
    max_iter::Int = 1000,
    nargs...)
  @assert max_iter > 0
  X = features(data)
  t = targets(data)
  b = bias(data)
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
  model._native_model = f
  model._coefficients = theta
  model._cost = cost(model, data)

  # define callback
  function cb(iter, theta, cost, grad)
    model._coefficients = theta
    model._cost = cost
    callback()
  end

  # Fit model
  r = Regression.solve!(Regression.GD(), f, theta,
                        Regression.Options(maxiter = max_iter, nargs...),
                        cb)

  model._coefficients = r.sol
  model._cost = r.fval
  r.niters
end

function train!{E<:ClassEncoding, G<:Any}(
    callback::Function,
    model::LogisticRegression,
    data::EncodedInMemoryLabeledDataSource{E,G,1},
    solver::Solver.GradientDescent,
    args...;
    nargs...)
  if model.l1_coef == model.l2_coef == 0.
    # No reg
    newModel = EmpiricalRiskLearner(LogisticLoss(), EmpiricalRisks.ZeroReg(), box=false)
    model._risk_learner = newModel
    train!(callback, newModel, data, solver, args...; nargs...)
  elseif model.l1_coef == 0. && model.l2_coef != 0.
    # Only l2 reg
    newModel = EmpiricalRiskLearner(LogisticLoss(), EmpiricalRisks.SqrL2Reg(model.l2_coef), box=false)
    model._risk_learner = newModel
    train!(callback, newModel, data, solver, args...; nargs...)
  elseif model.l1_coef != 0. && model.l2_coef == 0.
    # Only l1 reg
    newModel = EmpiricalRiskLearner(LogisticLoss(), EmpiricalRisks.L1Reg(model.l1_coef), box=false)
    model._risk_learner = newModel
    train!(callback, newModel, data, solver, args...; nargs...)
  else
    throw(ArgumentError("Hyperparameter of LogisticRegression have an illegal combination"))
  end
end
