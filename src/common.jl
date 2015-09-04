export AbstractLearner, AbstractClassifier

abstract AbstractLearner
abstract AbstractClassifier <: AbstractLearner

function safeRound(num)
  if VERSION < v"0.4-"
    iround(num)
  else
    round(Integer, num)
  end
end

function safeFloor(num)
  if VERSION < v"0.4-"
    ifloor(num)
  else
    floor(Integer, num)
  end
end
