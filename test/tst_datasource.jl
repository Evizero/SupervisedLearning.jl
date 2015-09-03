
#-----------------------------------------------------------

msg("DataSource: class hierachy")

@test LabeledDataSource <: DataSource
@test InMemoryLabeledDataSource <: LabeledDataSource
@test EncodedInMemoryLabeledDataSource <: InMemoryLabeledDataSource

#-----------------------------------------------------------

msg("DataSource: abstract methods standard implementation")

immutable FaultyDataSource <: LabeledDataSource
end

source = FaultyDataSource()
@test_throws ArgumentError nobs(source)
@test_throws ArgumentError features(source)
@test_throws ArgumentError features(source, 1, 2)
@test_throws ArgumentError targets(source)
@test_throws ArgumentError targets(source, 1, 2)
@test_throws ArgumentError bias(source)

#-----------------------------------------------------------

msg("LabeledDataSource: interface stability")

X = [1. 2. 3.;
     4. 5. 6.]
w = [1.,2.]
wn = [1. 2.;
      3. 4.]
ce1 = MultivalueClassEncoding(["V1","V2"])
ce2 = OneOfKClassEncoding(["V1","V2"])
@test_throws DimensionMismatch EncodedInMemoryLabeledDataSource(X, w, ce1)
@test_throws ArgumentError EncodedInMemoryLabeledDataSource(X, w, ce2)
@test_throws DimensionMismatch EncodedInMemoryLabeledDataSource(X, wn, ce2)
@test_throws MethodError EncodedInMemoryLabeledDataSource(X, wn, ce1)

#-----------------------------------------------------------

msg("LabeledDataSource: constructors")

X = [1. 2. 3.;
     4. 5. 6.]
t = [1., 2., 3.]
tn = [1. 2. 3.;
      4. 5. 6.;
      7. 8. 9.;
      1. 2. 3.]

ce = MultivalueClassEncoding(["V1","V2","V3"])
ds = EncodedInMemoryLabeledDataSource(X, t, ce)
@test nobs(ds) == 3
@test nvar(ds) == 2
@test bias(ds) == 1.
@test features(ds) == X
@test features(ds, 1, 1) == [1. 4.]'
@test features(ds, 2, 2) == [2. 3.; 5. 6.]
@test targets(ds, 1, 1) == [1.]
@test targets(ds, 2, 2) == [2., 3]
@test nclasses(ds) == 3
@test labels(ds) == ["V1","V2","V3"]

ds = dataSource(X, t, ce, bias=0.)
@test typeof(ds) <: EncodedInMemoryLabeledDataSource
@test nobs(ds) == 3
@test nvar(ds) == 2
@test bias(ds) == 0.
@test features(ds) == X
@test features(ds, 1, 1) == [1. 4.]'
@test features(ds, 2, 2) == [2. 3.; 5. 6.]
@test targets(ds, 1, 1) == [1.]
@test targets(ds, 2, 2) == [2., 3]
@test nclasses(ds) == 3
@test labels(ds) == ["V1","V2","V3"]

ce = OneOfKClassEncoding(["V1","V2","V3","V4"])
ds = EncodedInMemoryLabeledDataSource(X, tn, ce, .3)
@test nobs(ds) == 3
@test nvar(ds) == 2
@test bias(ds) == .3
@test features(ds) == X
@test features(ds, 1, 1) == [1. 4.]'
@test features(ds, 2, 2) == [2. 3.; 5. 6.]
@test targets(ds) == tn
@test targets(ds, 1, 1) == [1. 4. 7. 1.]'
@test targets(ds, 2, 2) == [2. 3.; 5. 6.; 8. 9.; 2. 3.]
@test nclasses(ds) == 4
@test labels(ds) == ["V1","V2","V3","V4"]

ds = dataSource(X, tn, ce, bias=0.)
@test typeof(ds) <: EncodedInMemoryLabeledDataSource
@test nobs(ds) == 3
@test nvar(ds) == 2
@test bias(ds) == 0.
@test features(ds) == X
@test features(ds, 1, 1) == [1. 4.]'
@test features(ds, 2, 2) == [2. 3.; 5. 6.]
@test targets(ds) == tn
@test targets(ds, 1, 1) == [1. 4. 7. 1.]'
@test targets(ds, 2, 2) == [2. 3.; 5. 6.; 8. 9.; 2. 3.]
@test nclasses(ds) == 4
@test labels(ds) == ["V1","V2","V3","V4"]

#-----------------------------------------------------------

msg("LabeledDataSource: splitTrainTest")

X = rand(10, 100)
t = round(rand(100) * 4) + 1
ce = MultivalueClassEncoding(["V1","V2","V3","V4","V5"])
cen = OneOfKClassEncoding(["V1","V2","V3","V4","V5"])
tn = labelencode(cen, labeldecode(ce, t))

ds = EncodedInMemoryLabeledDataSource(X, t, ce)
trainDs, testDs = splitTrainTest!(ds, p_train = .7)
@test typeof(trainDs) <: EncodedInMemoryLabeledDataSource
@test typeof(testDs) <: EncodedInMemoryLabeledDataSource
@test nobs(trainDs) + nobs(testDs) == nobs(ds)

ds = EncodedInMemoryLabeledDataSource(X, tn, cen)
trainDs, testDs = splitTrainTest!(ds, p_train = .7)
@test typeof(trainDs) <: EncodedInMemoryLabeledDataSource
@test typeof(testDs) <: EncodedInMemoryLabeledDataSource
@test nobs(trainDs) + nobs(testDs) == nobs(ds)

#-----------------------------------------------------------

msg("LabeledDataSource: DataFrames")

#using RDatasets
#data = dataset("datasets", "mtcars")
#formula = AM ~ DRat + WT
#ds = dataSource(formula, data)
#@test bias(ds) == 1.
