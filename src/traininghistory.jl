export TrainingHistory
export @push!, @enumerate, @get, @length

using DataStructures

import Base: length, push!, get, enumerate

# ==========================================================================

if VERSION < v"0.4-"
  typealias TupleType{T,V} (T, V)
else
  typealias TupleType{T,V} Tuple{T, V}
end

type TrainingHistory{T<:Integer}
  _storage::Dict{Symbol, Queue}
end

function TrainingHistory{T<:Integer}(::Type{T} = Int64)
  TrainingHistory{T}(Dict{Symbol, Queue}())
end

# ==========================================================================
# Functions

function length{T<:Integer}(history::TrainingHistory{T}, key::Symbol)
  length(history._storage[key])
end

function push!{T<:Integer,V<:Any}(history::TrainingHistory{T},
                                  iteration::T,
                                  key::Symbol,
                                  value::V)
  lastiter = zero(T)
  if !haskey(history._storage, key)
    iteration >= lastiter || throw(ArgumentError("Iterations must be greater than or equal to 0"))
    history._storage[key] = Queue(TupleType{T,V})
  else
    lastiter, _ = back(history._storage[key])
    iteration > lastiter || throw(ArgumentError("Iterations must increase over time"))
  end
  enqueue!(history._storage[key], (iteration, value))
  value
end

function enumerate{T<:Integer}(history::TrainingHistory{T}, key::Symbol)
  history._storage[key].store
end

function get{T<:Integer}(history::TrainingHistory{T}, key::Symbol)
  l = length(history, key)
  k, v = front(history._storage[key])
  karray = zeros(T, l)
  varray = Array(typeof(v), l)
  i = 1
  for (k, v) in enumerate(history, key)
    karray[i] = k
    varray[i] = v
    i += 1
  end
  karray, varray
end

# ==========================================================================
# Convenience macros

macro push!(history, iteration, func)
  key = string(func)
  esc(:(push!($history, $iteration, symbol($key), $func)))
end

macro enumerate(history, func)
  key = string(func)
  esc(:(enumerate($history, symbol($key))))
end

macro get(history, func)
  key = string(func)
  esc(:(get($history, symbol($key))))
end

macro length(history, func)
  key = string(func)
  esc(:(length($history, symbol($key))))
end
