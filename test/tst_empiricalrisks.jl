using SupervisedLearning

using RDatasets
using UnicodePlots

dataFrame = dataset("datasets", "mtcars")
formula = AM ~ DRat + WT

data = dataSource(formula, dataFrame, SignedClassEncoding)
model = Classifier.LogisticRegression(l2_coef = .001)
solver = Solver.GradientDescent()

train!(model, data, solver, max_iter = 1000, break_every = 10) do
  #println(model._risk_learner._iterations)
end

print(lineplot(trainingCurve(model)...))


d = 3
n = 1000
w = randn(d+1)
X = randn(d, n)
t = sign(X'w[1:d] + w[d+1] + 0.05 * randn(n))

data = dataSource(X, t, SignedClassEncoding(["no","yes"]))
model = Classifier.LogisticRegression(l2_coef = .001)
solver = Solver.GradientDescent()

train!(model, data, solver, max_iter = 1000, break_every = 10) do
  #println(model._risk_learner._iterations)
end

print(lineplot(trainingCurve(model)...))
