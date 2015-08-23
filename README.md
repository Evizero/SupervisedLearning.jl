# SupervisedLearning

Work in progress for a front-end supervised learning framework.

[![Build Status](https://travis-ci.org/Evizero/SupervisedLearning.jl.svg?branch=master)](https://travis-ci.org/Evizero/SupervisedLearning.jl)

## Planned API

```Julia
using SupervisedLearning
using RDatasets

data = dataset("datasets", "mtcars")

# In this case the dataset will be in-memory
# There will also be support for datastreaming from HDF5
problemSet = dataSource(AM ~ DRat + WT, data)

# doesn't actually copy the data, but indicies
trainSet, testSet = splitTrainTest(problemSet, p_train = .75)

# Specifies the model and modelspecific parameter
model = Classifier.LogisticRegression(l1_coef=0.001)

# Backend for neural networks will be Mocha.jl
# model = Classifier.FeedForwardNeuralNetwork([4,5,7],[ReLu,ReLu,ReLu])

# train! mutates the model state
#  * the do-block is the callback function which also allows for early stopping
#  * In this case :l_bfgs will result in using Optim.jl as backend
#  * There will also be stochastic gradient descent with minibatches
train!(model, trainSet, method=:l_bfgs, max_iter = 1000, break_every = 100) do
  x,y = trainingCurve(model)
  # integrated with UnicodePlots.jl for working in the REPL
  print(lineplot(x, y))
end

ŷ = predict(model, testSet)
```
