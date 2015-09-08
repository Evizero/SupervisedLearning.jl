export State, StateError
export @stateidx

import Base: <, >, ==, !=

# ==========================================================================
# Constants for the State

const STATES = [:error, :uninitialized, :initialized, :training, :trystop, :trained, :converged]
const STATES_LOOKUP = Dict{Symbol,Int}()
for i in 1:length(STATES)
  STATES_LOOKUP[STATES[i]] = i
end

macro stateidx(state)
  idx = STATES_LOOKUP[eval(state)]
  :($idx)
end

# ==========================================================================
# State type for convenience and comparison 

immutable State
  _idx::Int
  _symb::Symbol

  State(idx::Int) = new(idx, STATES[idx])
  State(symb::Symbol) = new(STATES_LOOKUP[symb], symb)
end

Base.show(io::IO, state::State) = print(io, state._idx, ": ", state._symb)
Base.convert(::Type{State}, idx::Int) = State(idx)
Base.convert(::Type{State}, symb::Symbol) = State(symb)

for op = (:<, :>, :(==), :(!=), :(<=), :(>=))
  @eval begin
    ($op)(s1::State, s2::State) = ($op)(s1._idx, s2._idx)
    ($op)(s::State,  i::Int)    = ($op)(s._idx, i)
    ($op)(i::Int,    s::State)  = ($op)(i, s._idx)
    ($op)(s::State,  k::Symbol) = ($op)(s._idx, STATES_LOOKUP[k])
    ($op)(k::Symbol, s::State)  = ($op)(STATES_LOOKUP[k], s._idx)
  end
end

# ==========================================================================
# State exception used to signal that a function was called in an illegal state

type StateError <: Exception
  msg::String
end
