using SupervisedLearning
using UnicodePlots

import EmpiricalRisks
import Regression

d = 10
n = 100
p = 3
w = randn(d+1)
Xtr = randn(d, n)
Xte = randn(d, n)
for i = 2:d
  Xtr[i,:] =  Xtr[1,:] .^ i
  Xtr[i,:] = (Xtr[i,:] - mean(Xtr[i,:])) ./ std(Xtr[i,:])
  Xte[i,:] =  Xte[1,:] .^ i
  Xte[i,:] = (Xte[i,:] - mean(Xte[i,:])) ./ std(Xte[i,:])
end
tr = sign(Xtr[1:p,:]'w[1:p] + w[d+1] + 1.5 * randn(n))
te = sign(Xte[1:p,:]'w[1:p] + w[d+1])

trainSet = dataSource(Xtr, tr, SignedClassEncoding(["no","yes"]))
testSet = dataSource(Xte, te, SignedClassEncoding(["no","yes"]))

print(barplot(classDistribution(trainSet)..., title = "Trainset", width = 30))
print(barplot(classDistribution(testSet)..., title = "Testset", width = 30))

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
  maxiter = 1000
  train!(model, trainSet, solver, max_iter = maxiter, break_every = 5) do
    @test state(model) == :training
    @test iterations(model) % 5 == 0
    remember!(model, :testSet, cost(model, testSet))
  end

  @test state(model) == :trained
  @test 0 < iterations(model) <= maxiter
  @test_approx_eq cost(model) -fitness(model)
  @test_approx_eq cost(model, trainSet) -fitness(model, trainSet)
  @test_approx_eq cost(model) cost(model, trainSet)

  x, y = trainingCurve(model)#history(model, :tst)
  x, y = history(model, :testSet)
  plt = lineplot(x, y, ylim=[floor(minimum(y)), ceil(maximum(y))], xlim=[0, maxiter], width = 30, height = 2)
  annotate!(plt, :r, 1, "$(name(model)) (test acc: $(round(accuracy(model, testSet),2)))")
  annotate!(plt, :r, 2, "$(typeof(solver)) $desc")
  annotate!(plt, :bl, ""); annotate!(plt, :br, "")
  print(plt)

  y = predict(model, trainSet)
  #@test sum(y .== groundtruth(trainSet)) > .8 * nobs(trainSet)
  @test_approx_eq accuracy(model, trainSet) (sum(y .== groundtruth(trainSet)) / length(groundtruth(trainSet)))

  p = predictProb(model, trainSet)
  @test length(y) == length(p)
  @test maximum(p) <= 1.
  @test minimum(p) >= 0.
end
