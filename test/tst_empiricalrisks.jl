using SupervisedLearning
using UnicodePlots

import EmpiricalRisks
import Regression

#using RDatasets

# dataFrame = dataset("datasets", "mtcars")
# formula = AM ~ DRat + WT

# data = dataSource(formula, dataFrame, SignedClassEncoding)
# model = Classifier.LogisticRegression(l2_coef = .001)
# solver = Solver.GradientDescent()

# train!(model, data, solver, max_iter = 1000, break_every = 10) do
#   #println(model._risk_learner._iterations)
# end

# print(lineplot(trainingCurve(model)...))


d = 3
n = 1000
w = randn(d+1)
X = randn(d, n)
t = sign(X'w[1:d] + w[d+1] + 0.0001 * randn(n))

data = dataSource(X, t, SignedClassEncoding(["no","yes"]))
model = Classifier.LogisticRegression(l2_coef = .01)
solver = Solver.GradientDescent()

println("State: $(state(model))")
println("Iterations: $(iterations(model))")
println("Cost: $(cost(model))")

train!(model, data, solver, max_iter = 500, break_every = 100) do
  println("Iterations: $(iterations(model))")
  println("Cost: $(cost(model))")
end

# train!(model, data, solver, max_iter = 1000, break_every = 100) do
#   println("Iterations: $(iterations(model))")
# end
#lineplot(trainingCurve(model)...)
println()
print(lineplot(trainingCurve(model)...))
println("State: $(state(model))")
println("Iterations: $(iterations(model))")
println("Fitness: $(fitness(model))")
println("Cost: $(cost(model))")
#@test_approx_eq w model._risk_learner._coefficients

# f1 = Regression.RegRiskFun(EmpiricalRisks.riskmodel(EmpiricalRisks.AffinePred(d, 1.), EmpiricalRisks.LogisticLoss()), 
#                           Regression.SqrL2Reg(1.0e-2),
#                           X, t)
# theta = zeros(d+1)

# @time Regression.solve!(Regression.GD(), f1, theta,
#                     Regression.Options(maxiter = 100),
#                     Regression.no_op)
# @time Regression.solve!(Regression.GD(), f1, theta,
#                     Regression.Options(maxiter = 10000),
#                     Regression.no_op)

# using EmpiricalRisks
# using ArrayViews

# pm = AffinePred(d, 1.)
# @time s = predict(pm, w, X)
# @time s = predict(pm, w, X)

# r = zeros(n)
# @time predict!(r, pm, w, X)
# @time predict!(r, pm, w, X)

# println("Loop with predict")
# @time for i = 1:1000
#   s = predict(pm, w, X)
# end

# println("Loop with predict!")
# @time for i = 1:1000
#   s = predict!(r, pm, w, X)
# end

# s = predict(pm, w, X)
# @test_approx_eq s r

# println("Risk models")
# rmodel = riskmodel(pm, LogisticLoss())
# value(rmodel, w, X, t)
# @time value(rmodel, w, X, t)

# rmodel = riskmodel(pm, LogisticLoss())
# value!(r, rmodel, w, X, t)
# @time value!(r, rmodel, w, X, t)

# println("Loop with value")
# @time for i = 1:100
#   s = value(rmodel, w, X, t)
# end

# println("Loop with value!")
# @time for i = 1:100
#   s = value!(r, rmodel, w, X, t)
# end

# data = dataSource(X, t, SignedClassEncoding(["no","yes"]))
# model = Classifier.LogisticRegression(l2_coef = .001)
# solver = Solver.GradientDescent()

# println("State: $(state(model))")
# println("Iterations: $(iterations(model))")
# println("Cost: $(cost(model))")

# train!(model, data, solver, max_iter = 1000, break_every = 100)
# @time train!(model, data, solver, max_iter = 1000, break_every = 100)

# # train!(model, data, solver, max_iter = 1000, break_every = 100) do
# #   println("Iterations: $(iterations(model))")
# # end
# #lineplot(trainingCurve(model)...)
# #@time print(lineplot(trainingCurve(model)...))
# println("State: $(state(model))")
# println("Iterations: $(iterations(model))")
# println("Fitness: $(fitness(model))")
# println("Cost: $(cost(model))")

