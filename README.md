# SupervisedLearning

Work in progress for a front-end supervised learning framework.

[![Build Status](https://travis-ci.org/Evizero/SupervisedLearning.jl.svg?branch=master)](https://travis-ci.org/Evizero/SupervisedLearning.jl)

The goal of this library is manyfold:

- **Education:** allow the user to play around with the models, solvers, etc. for educational purposes. Provide a good base for course exercises. For example visualizing the learning curve of neural networks using different optimization algorithms.
- **Research:** Swap out parts of the machine learning pipeline with custom implementations without losing the ability to utilize the rest of the framework. For example to prototype new prediction models.
- **Application:** Porcelain interface to apply machine learning to given datasets in a convenient way. There might be multiple high-level interface for different usergroups (e.g. one that mimics R's caret package)

## Planned High-level API

The following code should allready work

```Julia
using SupervisedLearning
using RDatasets
using UnicodePlots

data = dataset("datasets", "mtcars")

# In this case the dataset will be in-memory and encoded to -1, 1
# There will also be support for datastreaming from HDF5
problemSet = dataSource(AM ~ DRat + WT + DRat&WT, data, SignedClassEncoding)

# Convenient to use with UnicodePlots
print(barplot(classDistribution(problemSet)...))

# Methods for splitting the abstract data sets
trainSet, testSet = splitTrainTest!(problemSet, p_train = .75)

# Specifies the model and modelspecific parameter
model = Classifier.LogisticRegression(l2_coef=0.1)

# Backend for neural networks will be Mocha.jl or OnlineAI.jl
# model = Classifier.FeedForwardNeuralNetwork([4,5,7],[ReLu,ReLu,ReLu])

# train! mutates the model state
#  * the do-block is the callback function which also allows for early stopping
#  * In the regression case Solver.GradientDescent() will result in using Regression.jl, 
#    otherwise (in most deterministic cases) Optim.jl
#  * There will also be stochastic gradient descent with minibatches
train!(model, trainSet, Solver.GradientDescent(), max_iter = 10000, break_every = 100) do

  # You can easily store custom learning curves or other arbitrary values
  # They will be linked to the associated iteration automatically
  remember!(model, :testsetLoss, cost(model, testSet))
  
  # You can also use the callback to execute any code
  # For example to print informative messages
  println("Testset accuracy: ", accuracy(model, testSet))
end

# The loss of the training set is stored by default and can be accessed with trainingCurve
# x is a Vector{Int} of iterations with stepsize break_every,
# y is a Vector{Float64} where y[i] is the trainSet cost at x[i]
x, y = trainingCurve(model)
print(lineplot(x, y, title = "Learning curve for trainSet"))

# Customly stored curves can be accessed with "history"
# x is a Vector{Int} of iterations (exact values depend on when you called remember!),
# y is a Vector{Float64} where y[i] is the trainSet cost at x[i]
x, y = history(model, :testsetLoss)
print(lineplot(x, y, title = "Learning curve for testSet"))

ŷ = predict(model, testSet) # what the model says
t = groundtruth(testSet) # what it should be
```

## Planned Mid-level API

This is just a rough draft and still object to change

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
trainSet, testSet = splitTrainTest!(problemSet, p_train = .75)

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
print(barplot(accuracy(gsResult, testSet)...))

# Get the best model
bestModel = gsResult.bestModel
ŷ = predict(bestModel, testSet)
```
