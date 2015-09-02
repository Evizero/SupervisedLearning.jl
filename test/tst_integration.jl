
using SupervisedLearning
using RDatasets
using UnicodePlots

data = dataset("datasets", "mtcars")
formula = AM ~ DRat + WT

#-----------------------------------------------------------

msg("Encoding dataframe per hand")

mf = ModelFrame(formula, data)
mm = ModelMatrix(mf)
dfBias = in(0,mm.assign)
X = dfBias ? mm.m[:,2:end]: mm.m
extBias = dfBias * 1.0
t = convert(Vector{Float64}, model_response(mf))

ce = ZeroOneClassEncoding(["No", "Yes"])

ds = dataSource(X', t, ce, bias=extBias)
@test typeof(ds) <: EncodedInMemoryLabeledDataSource

#-----------------------------------------------------------

msg("Encoding dataframe with dataSource")

ds = dataSource(formula, data, SignedClassEncoding)
@test typeof(ds) <: EncodedInMemoryLabeledDataSource

#-----------------------------------------------------------

msg("Encoding dataframe with encodeDataSource")

ds = dataSource(formula, data)
@test typeof(ds) <: DataFrameLabeledDataSource
ds = encodeDataSource(ds, ZeroOneClassEncoding)
@test typeof(ds) <: EncodedInMemoryLabeledDataSource

#barplot(classDistribution(ds)...)
