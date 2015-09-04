# SupervisedLearning

Work in progress for a front-end supervised learning framework.

[![Build Status](https://travis-ci.org/Evizero/SupervisedLearning.jl.svg?branch=master)](https://travis-ci.org/Evizero/SupervisedLearning.jl)

The goal of this library is manyfold:

- **Education:** allow the user to play around with the models, solvers, etc. for educational purposes. Provide a good base for course exercises. For example visualizing the learning curve of neural networks using different optimization algorithms.
- **Research:** Swap out parts of the machine learning pipeline with custom implementations without losing the ability to utilize the rest of the framework. For example to prototype new prediction models.
- **Application:** Porcelain interface to apply machine learning to given datasets in a convenient way. There might be multiple high-level interface for different usergroups (e.g. one that mimics R's caret package)

## Planned High-level API

this is just a rough draft and object to change

```Julia
using SupervisedLearning
using RDatasets

data = dataset("datasets", "mtcars")

# In this case the dataset will be in-memory and encoded to -1, 1
# There will also be support for datastreaming from HDF5
problemSet = dataSource(AM ~ DRat + WT, data, SignedClassEncoding)

# Convenient to use with UnicodePlots
print(barplot(classDistribution(problemSet)...))

# Methods for splitting the abstract data sets
trainSet, testSet = splitTrainTest(problemSet, p_train = .75)

# Specifies the model and modelspecific parameter
model = Classifier.LogisticRegression(l2_coef=0.001)

# Backend for neural networks will be Mocha.jl
# model = Classifier.FeedForwardNeuralNetwork([4,5,7],[ReLu,ReLu,ReLu])

# train! mutates the model state
#  * the do-block is the callback function which also allows for early stopping
#  * In this case L_BFGS() will result in using Optim.jl with :l_bfgs as backend
#  * There will also be stochastic gradient descent with minibatches
train!(model, trainSet, solver = Solver.L_BFGS(), max_iter = 1000, break_every = 100) do
  x,y = trainingCurve(model)
  # integrated with UnicodePlots.jl for working in the REPL
  print(lineplot(x, y))
  println("Testset accuracy: ", accuracy(model, testSet))
end

ŷ = predict(model, testSet)
```

## Planned Mid-level API

```Julia
using SupervisedLearning
using RDatasets

data = dataset("datasets", "mtcars")

# In this case the dataset will be in-memory.
# Specifying the encoding is not necessary.
# The model will select the encoding it needs automatically
# Trees for example don't need an encoding at all.
problemSet = dataSource(AM ~ DRat + WT, data)

# Methods for splitting the abstract data sets
trainSet, testSet = splitTrainTest(problemSet, p_train = .75)

# Perform a gridsearch over an arbitrary modelspace
gsResult = gridsearch([.001, .01, .1], [.0001, .0003]) do lr, lambda

  # Perform cross validation to get a good estimate for the hyperparameter performance
  cvResult = crossvalidate(trainSet, k = 5) do trainFold, valFold

    # Specify the model and model-specific parameters
    model = Classifier.LogisticRegression(l2_coef = lambda)

    # Specify the solver and solver-specific parameters
    solver = Solver.NaiveGradientDescent(learning_rate = lr, normalize_gradient = false)

    # train! mutates the model state
    train!(model, trainFold, solver, max_iter = 1000)

    # make sure to return the trained model
    model
  end

  # You can return a model or a cvResult to gridsearch
  cvResult
end

# Plot the final accuracy of all trained models using UnicodePlots
print(barplot(accuracy(result, testSet)...))

# Get the best model
bestModel = result.bestModel
ŷ = predict(bestModel, testSet)
```
