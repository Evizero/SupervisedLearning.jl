export TrainingHistory
export @push!, iterate

using DataStructures

import Base: length, push!, get

# ==========================================================================

if VERSION < v"0.4-"
  typealias TupleType{T} (T, Any)
else
  typealias TupleType{T} Tuple{T, Any}
end

type TrainingHistory{T<:Integer}
  _storage::Dict{Function, Queue{Deque{TupleType{T}}}}
end

function TrainingHistory{T<:Integer}(::Type{T} = Int64)
  TrainingHistory(Dict{Function, Queue{Deque{TupleType{T}}}}())
end

# ==========================================================================
# Functions

function length{T<:Integer}(history::TrainingHistory{T}, f::Function)
  length(history._storage[f])
end

function push!{T<:Integer}(history::TrainingHistory{T},
                           iteration::T,
                           f::Function,
                           args...; nargs...)
  res = f(args...; nargs...)
  lastiter = zero(T)
  if !haskey(history._storage, f)
    iteration >= lastiter || throw(ArgumentError("Iterations must not decrease over time"))
    history._storage[f] = Queue(TupleType{T})
  else
    lastiter, _ = back(history._storage[f])
    iteration >= lastiter || throw(ArgumentError("Iterations must not decrease over time"))
  end
  enqueue!(history._storage[f], (iteration, res))
  res
end

macro push!(history, iteration, func)
  esc(:(push!($history, $iteration, $(func.args...))))
end

function iterate{T<:Integer}(history::TrainingHistory{T}, f::Function)
  history._storage[f].store
end

function get{T<:Integer}(history::TrainingHistory{T}, f::Function)
  l = length(history, f)
  karray = zeros(T, l)
  varray = Array(Any, l)
  i = 1
  for (k, v) in iterate(history, f)
    karray[i] = k
    varray[i] = v
    i += 1
  end
  karray, varray
end
