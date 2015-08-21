
#-----------------------------------------------------------

msg("DataSource: class hierachy")

@test LabeledDataSource <: DataSource
@test InMemoryLabeledDataSource <: LabeledDataSource

#-----------------------------------------------------------

msg("DataSource: abstract methods standard implementation")

immutable FaultyDataSource <: LabeledDataSource
end

source = FaultyDataSource()
@test_throws MethodError nobs(source)
@test_throws MethodError features(source)
@test_throws MethodError features(source, 1, 2)
@test_throws MethodError targets(source)
@test_throws MethodError targets(source, 1, 2)

#-----------------------------------------------------------

msg("LabeledDataSource: interface stability")

X = [1. 2. 3.;
     4. 5. 6.]
w = [1.,2.]
wn = [1. 2.;
      3. 4.]
@test_throws DimensionMismatch InMemoryLabeledDataSource(X, w)
@test_throws DimensionMismatch InMemoryLabeledDataSource(X, wn)

#-----------------------------------------------------------

msg("LabeledDataSource: constructors")

X = [1. 2. 3.;
     4. 5. 6.]
t = [1., 2., 3.]
tn = [1. 2. 3.;
      4. 5. 6.;
      7. 8. 9.;
      1. 2. 3.]

ds = InMemoryLabeledDataSource(X, t)
@test nobs(ds) == 3
@test nvar(ds) == 2
@test features(ds) == X
@test features(ds, 1, 1) == [1. 4.]'
@test features(ds, 2, 2) == [2. 3.; 5. 6.]
@test targets(ds) == t

ds = InMemoryLabeledDataSource(X, tn)
@test nobs(ds) == 3
@test nvar(ds) == 2
@test features(ds) == X
@test features(ds, 1, 1) == [1. 4.]'
@test features(ds, 2, 2) == [2. 3.; 5. 6.]
@test targets(ds) == tn
