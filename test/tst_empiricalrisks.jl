using SupervisedLearning
using UnicodePlots

import EmpiricalRisks
import Regression

d = 5
n = 1000
w = randn(d+1)
X = randn(d, n)
t = sign(X'w[1:d] + w[d+1] + 0.2 * randn(n))

data = dataSource(X, t, SignedClassEncoding(["no","yes"]))
print(barplot(classDistribution(data)..., width = 30))

trainSet, testSet = splitTrainTest!(data, p_train = .8)

models = [("without Regularization", Classifier.LogisticRegression()), 
          ("with L1 penalty", Classifier.LogisticRegression(l1_coef = .1)), 
          ("with L2 penalty", Classifier.LogisticRegression(l2_coef = .1))]

solvers = [Solver.GradientDescent()]

for solver in solvers, (desc, model) in models

  @test state(model) == :uninitialized
  @test iterations(model) == 0
  @test_throws StateError trainingCurve(model)
  @test_throws StateError cost(model)

  #train!(model, data, solver, max_iter = 50, break_every = 5)
  train!(model, trainSet, solver, max_iter = 50, break_every = 5) do
    @test state(model) == :training
    @test iterations(model) % 5 == 0
  end

  @test state(model) == :trained
  @test 0 < iterations(model) <= 500
  @test_approx_eq cost(model) -fitness(model)
  @test_approx_eq cost(model, trainSet) -fitness(model, trainSet)
  @test_approx_eq cost(model) cost(model, trainSet)

  x, y = trainingCurve(model)
  plt = lineplot(x, y, ylim=[floor(minimum(y)), ceil(maximum(y))], width = 30, height = 2)
  annotate!(plt, :r, 1, "$(name(model)) (test acc: $(round(accuracy(model, testSet),2)))")
  annotate!(plt, :r, 2, "$(typeof(solver)) $desc")
  annotate!(plt, :bl, ""); annotate!(plt, :br, "")
  print(plt)

  y = predict(model, trainSet)
  @test sum(y .== groundtruth(trainSet)) > .8 * nobs(trainSet)
  @test_approx_eq accuracy(model, trainSet) (sum(y .== groundtruth(trainSet)) / length(groundtruth(trainSet)))

  p = predictProb(model, trainSet)
  @test length(y) == length(p)
  @test maximum(p) <= 1.
  @test minimum(p) >= 0.
end
