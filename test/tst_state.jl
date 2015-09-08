using SupervisedLearning

#-----------------------------------------------------------

symbols = [:error, :uninitialized, :initialized, :training, :trystop, :trained, :converged]
@test SupervisedLearning.STATES == symbols

#-----------------------------------------------------------

msg("State: constructors")

@test_throws MethodError State(1, :error)
@test_throws KeyError State(:foo)
@test_throws BoundsError State(0)
@test_throws BoundsError State(8)

#-----------------------------------------------------------

msg(string("State: execute show: ", State(2)))

#-----------------------------------------------------------

msg("State: construction from symbols")

for sym in symbols
  @test State(sym) == convert(State, sym)
  @test State(sym) == State(sym)
  @test State(sym) <= State(sym)
  @test State(sym) >= State(sym)
  @test State(sym) == sym
  @test State(sym) <= sym
  @test State(sym) >= sym
  @test sym == State(sym)
  @test sym <= State(sym)
  @test sym >= State(sym)
end

#-----------------------------------------------------------

msg("State: construction from numbers")

for i in 1:length(symbols)
  @test State(i) == convert(State, i)
  @test State(i) == State(i)
  @test State(i) <= State(i)
  @test State(i) >= State(i)
  @test State(i) == i
  @test State(i) <= i
  @test State(i) >= i
  @test i == State(i)
  @test i <= State(i)
  @test i >= State(i)
end

#-----------------------------------------------------------

msg("State: comparing with numbers and symbols")

for (i, sym) in enumerate(symbols)
  @test !(State(i) != State(sym))
  @test !(State(i) < State(sym))
  @test !(State(i) > State(sym))
  @test State(i) == State(sym)
  @test State(i) <= State(sym)
  @test State(i) >= State(sym)
  @test !(State(sym) != State(i))
  @test !(State(sym) < State(i))
  @test !(State(sym) > State(i))
  @test State(sym) == State(i)
  @test State(sym) <= State(i)
  @test State(sym) >= State(i)
  @test !(State(i) != sym)
  @test !(State(i) < sym)
  @test !(State(i) > sym)
  @test State(i) == sym
  @test State(i) <= sym
  @test State(i) >= sym
  @test !(sym != State(i))
  @test !(sym < State(i))
  @test !(sym > State(i))
  @test sym == State(i)
  @test sym <= State(i)
  @test sym >= State(i)
  @test State(sym) == i
  @test State(sym) <= i
  @test State(sym) >= i
  @test !(State(sym) != i)
  @test !(State(sym) < i)
  @test !(State(sym) > i)
  @test sym == State(i)
  @test sym <= State(i)
  @test sym >= State(i)
  @test !(sym != State(i))
  @test !(sym < State(i))
  @test !(sym > State(i))
  
  for j = 1:(i-1)
    @test State(j) != sym
    @test State(j) < sym
    @test State(j) <= sym
    @test !(State(j) > sym)
    @test !(State(j) == sym)
    @test !(State(j) >= sym)
    @test sym != State(j)
    @test sym > State(j)
    @test sym >= State(j)
    @test !(sym < State(j))
    @test !(sym == State(j))
    @test !(sym <= State(j))
    @test State(sym) != j
    @test State(sym) > j
    @test State(sym) >= j
    @test !(State(sym) < j)
    @test !(State(sym) == j)
    @test !(State(sym) <= j)
    @test sym != State(j)
    @test sym > State(j)
    @test sym >= State(j)
    @test !(sym == State(j))
    @test !(sym < State(j))
    @test !(sym <= State(j))
  end
  
  for j = (i+1):length(symbols)
    @test State(j) != sym
    @test State(j) > sym
    @test State(j) >= sym
    @test !(State(j) < sym)
    @test !(State(j) == sym)
    @test !(State(j) <= sym)
    @test sym != State(j)
    @test sym < State(j)
    @test sym <= State(j)
    @test !(sym > State(j))
    @test !(sym == State(j))
    @test !(sym >= State(j))
    @test State(sym) != j
    @test State(sym) < j
    @test State(sym) <= j
    @test !(State(sym) > j)
    @test !(State(sym) == j)
    @test !(State(sym) >= j)
    @test sym != State(j)
    @test sym < State(j)
    @test sym <= State(j)
    @test !(sym == State(j))
    @test !(sym > State(j))
    @test !(sym >= State(j))
  end
end
