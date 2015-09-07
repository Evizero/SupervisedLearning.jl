using SupervisedLearning

#-----------------------------------------------------------

msg("TrainingHistory: Basic functions")

history = TrainingHistory()

function f(i, b; muh=10)
  @test b == "yo"
  @test muh == .3
  i
end

function f2(i, b; muh=10)
  @test b == "my"
  @test muh == .1
  i
end

@test_throws ArgumentError push!(history, -1, f, 10, "yo", muh = .3)

numbers = collect(0:2:100)
for i = numbers
  @test push!(history, i, f, i + 1, "yo", muh = .3) == i + 1
  @test @push!(history, i, f2(i - 1, "my", muh = .1)) == i - 1
end

for (i, v) in iterate(history, f)
  @test in(i, numbers)
  @test i + 1 == v
end

for (i, v) in iterate(history, f2)
  @test in(i, numbers)
  @test i - 1 == v
end

a1, a2 = get(history, f)
@test typeof(a1) <: Vector && typeof(a2) <: Vector
@test length(a1) == length(a2) == length(numbers) == length(history, f)
@test a1 + 1 == a2

a1, a2 = get(history, f2)
@test typeof(a1) <: Vector && typeof(a2) <: Vector
@test length(a1) == length(a2) == length(numbers) == length(history, f2)
@test a1 - 1 == a2

@test_throws ArgumentError push!(history, 10, f, 10, "yo", muh = .3)
@test_throws KeyError iterate(history, sign)
@test_throws KeyError length(history, sign)

#-----------------------------------------------------------

msg("TrainingHistory: Storing arbitrary types")

history = TrainingHistory(Uint8)

for i = 1:100
  @test push!(history, uint8(i), string, "i=", i + 1) == string("i=", i+1)
  @test push!(history, uint8(i), float, i + 1) == float(i+1)
end

a1, a2 = get(history, string)
@test typeof(a1) <: Vector{Uint8}
for entry in a2
  @test typeof(entry) <: String
end

a1, a2 = get(history, float)
@test typeof(a1) <: Vector{Uint8}
for entry in a2
  @test typeof(entry) <: FloatingPoint
end