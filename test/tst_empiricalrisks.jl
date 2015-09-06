using SupervisedLearning
using UnicodePlots

import EmpiricalRisks
import Regression

d = 3
n = 1000
w = randn(d+1)
X = randn(d, n)
t = sign(X'w[1:d] + w[d+1] + 0.2 * randn(n))

data = dataSource(X, t, SignedClassEncoding(["no","yes"]))

models = [Classifier.LogisticRegression(), 
          Classifier.LogisticRegression(l1_coef = .1), 
          Classifier.LogisticRegression(l2_coef = .1)]

for model in models
  solver = Solver.GradientDescent()

  @test state(model) == :untrained
  @test iterations(model) == 0
  @test_throws ArgumentError trainingCurve(model)

  #train!(model, data, solver, max_iter = 50, break_every = 5)
  train!(model, data, solver, max_iter = 50, break_every = 5) do
    state(model) == :training
    @test iterations(model) % 5 == 0
  end

  @test state(model) == :trained
  @test 0 < iterations(model) <= 50
  @test_approx_eq cost(model) -fitness(model)
  @test_approx_eq cost(model, data) -fitness(model, data)
  @test_approx_eq cost(model) cost(model, data)

  print(lineplot(trainingCurve(model)..., title = "Gradient Descent"))

  y = predict(model, data)
  @test sum(y .== groundtruth(data)) > .8 * n
  @test_approx_eq accuracy(model, data) (sum(y .== groundtruth(data)) / length(groundtruth(data)))

  p = predictProb(model, data)
  @test length(y) == length(p)
  @test maximum(p) <= 1.
  @test minimum(p) >= 0.
end
