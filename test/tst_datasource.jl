
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
@test_throws MethodError bias(source)

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
@test bias(ds) == 1.
@test features(ds) == X
@test features(ds, 1, 1) == [1. 4.]'
@test features(ds, 2, 2) == [2. 3.; 5. 6.]
@test targets(ds, 1, 1) == [1.]
@test targets(ds, 2, 2) == [2., 3]

ds = dataSource(X, t, bias=0.)
@test typeof(ds) == InMemoryLabeledDataSource{Float64,1}
@test nobs(ds) == 3
@test nvar(ds) == 2
@test bias(ds) == 0.
@test features(ds) == X
@test features(ds, 1, 1) == [1. 4.]'
@test features(ds, 2, 2) == [2. 3.; 5. 6.]
@test targets(ds, 1, 1) == [1.]
@test targets(ds, 2, 2) == [2., 3]

ds = InMemoryLabeledDataSource(X, tn)
@test nobs(ds) == 3
@test nvar(ds) == 2
@test bias(ds) == 1.
@test features(ds) == X
@test features(ds, 1, 1) == [1. 4.]'
@test features(ds, 2, 2) == [2. 3.; 5. 6.]
@test targets(ds) == tn
@test targets(ds, 1, 1) == [1. 4. 7. 1.]'
@test targets(ds, 2, 2) == [2. 3.; 5. 6.; 8. 9.; 2. 3.]

ds = dataSource(X, tn, bias=0.)
@test typeof(ds) == InMemoryLabeledDataSource{Float64,2}
@test nobs(ds) == 3
@test nvar(ds) == 2
@test bias(ds) == 0.
@test features(ds) == X
@test features(ds, 1, 1) == [1. 4.]'
@test features(ds, 2, 2) == [2. 3.; 5. 6.]
@test targets(ds) == tn
@test targets(ds, 1, 1) == [1. 4. 7. 1.]'
@test targets(ds, 2, 2) == [2. 3.; 5. 6.; 8. 9.; 2. 3.]

