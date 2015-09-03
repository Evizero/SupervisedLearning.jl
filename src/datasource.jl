export DataSource, LabeledDataSource, InMemoryLabeledDataSource
export EncodedInMemoryLabeledDataSource, DataFrameLabeledDataSource
export nobs, nvar, features, targets, bias, splitTrainTest
export dataSource, encodeDataSource

using ArrayViews
using DataFrames
import StatsBase.nobs

macro _not_implemented()
  quote
    throw(ArgumentError("Not implemented for the given datasource type"))
  end
end

# ==========================================================================

abstract DataSource

nobs(source::DataSource) = @_not_implemented
nvar(source::DataSource) = @_not_implemented
features(source::DataSource) = @_not_implemented
features(source::DataSource, offset::Int, length::Int) = @_not_implemented

# labeled sources

abstract LabeledDataSource <: DataSource
abstract InMemoryLabeledDataSource <: LabeledDataSource

targets(source::LabeledDataSource) = @_not_implemented
targets(source::LabeledDataSource, offset::Int, length::Int) = @_not_implemented
bias(source::LabeledDataSource) = @_not_implemented
nclasses(source::LabeledDataSource) = @_not_implemented
labels(source::LabeledDataSource) = @_not_implemented
classDistribution(source::LabeledDataSource) = @_not_implemented

# ==========================================================================
# In-memory labeled sources

immutable EncodedInMemoryLabeledDataSource{E<:ClassEncoding,N} <: InMemoryLabeledDataSource
  features::AbstractArray{Float64,2}
  targets::AbstractArray{Float64,N}
  encoding::E
  bias::Float64

  function EncodedInMemoryLabeledDataSource(features::AbstractArray{Float64,2},
                                            targets::AbstractArray{Float64,1},
                                            encoding::E,
                                            bias::Float64)
    (typeof(encoding) <: OneOfKClassEncoding) && throw(ArgumentError("Can't have OneOutOfK-Encoding with a one vector as target"))
    size(features,2) == length(targets) || throw(DimensionMismatch("Features and targets have to have the same number of observations"))
    new(features, targets, encoding, bias)
  end

  function EncodedInMemoryLabeledDataSource(features::AbstractArray{Float64,2},
                                            targets::AbstractArray{Float64,2},
                                            encoding::OneOfKClassEncoding,
                                            bias::Float64)
    nclasses(encoding) == size(targets,1) || throw(DimensionMismatch("Targets have to have the same number of rows as the encoding has labels"))
    size(features,2) == size(targets,2) || throw(DimensionMismatch("Features and targets have to have the same number of observations"))
    new(features, targets, encoding, bias)
  end
end

function EncodedInMemoryLabeledDataSource{E<:ClassEncoding}(features::AbstractArray{Float64,2},
                                          targets::AbstractArray{Float64,1},
                                          encoding::E,
                                          bias::Float64 = 1.)
   EncodedInMemoryLabeledDataSource{E,1}(features, targets, encoding, bias)
end

function EncodedInMemoryLabeledDataSource(features::AbstractArray{Float64,2},
                                          targets::AbstractArray{Float64,2},
                                          encoding::OneOfKClassEncoding,
                                          bias::Float64 = 1.)
   EncodedInMemoryLabeledDataSource{OneOfKClassEncoding,2}(features, targets, encoding, bias)
end

nobs{E<:ClassEncoding,N}(source::EncodedInMemoryLabeledDataSource{E,N}) = size(source.features, 2)
nvar{E<:ClassEncoding,N}(source::EncodedInMemoryLabeledDataSource{E,N}) = size(source.features, 1)
features{E<:ClassEncoding,N}(source::EncodedInMemoryLabeledDataSource{E,N}) = source.features
features{E<:ClassEncoding,N}(source::EncodedInMemoryLabeledDataSource{E,N}, offset::Int, length::Int) = view(source.features, :, offset:(offset+length-1))
targets{E<:ClassEncoding,N}(source::EncodedInMemoryLabeledDataSource{E,N}) = source.targets
targets{E<:ClassEncoding}(source::EncodedInMemoryLabeledDataSource{E,1}, offset::Int, length::Int) = view(source.targets, offset:(offset+length-1))
targets{E<:ClassEncoding}(source::EncodedInMemoryLabeledDataSource{E,2}, offset::Int, length::Int) = view(source.targets, :, offset:(offset+length-1))
bias{E<:ClassEncoding,N}(source::EncodedInMemoryLabeledDataSource{E,N}) = source.bias
nclasses{E<:ClassEncoding,N}(source::EncodedInMemoryLabeledDataSource{E,N}) = nclasses(source.encoding)
labels{E<:ClassEncoding,N}(source::EncodedInMemoryLabeledDataSource{E,N}) = labels(source.encoding)
classDistribution{E<:ClassEncoding,N}(source::EncodedInMemoryLabeledDataSource{E,N}) = classDistribution(source.encoding, labeldecode(source.encoding, source.targets))

function splitTrainTest{E<:ClassEncoding,N}(source::EncodedInMemoryLabeledDataSource{E,N};
                                            p_train = .7,
                                            balance_classes = false)
  X = source.features
  t = source.targets
  ce = source.encoding
  bias = source.bias
  n = nobs(source)
  sn = safeFloor(n * p_train)
  if N == 1
    trainIdx = collect(balance_classes ? StratifiedRandomSub(t, sn, 1): RandomSub(n, sn, 1))[1]
    testIdx = setdiff(1:n, trainIdx)
    trainData = dataSource(X[:, trainIdx], t[trainIdx], ce, bias = bias)
    testData = dataSource(X[:, testIdx], t[testIdx], ce, bias = bias)
    return trainData, testData
  else
    trainIdx = collect(balance_classes ? StratifiedRandomSub(labeldecode(t), sn, 1): RandomSub(n, sn, 1))[1]
    testIdx = setdiff(1:n, trainIdx)
    trainData = dataSource(X[:,trainIdx], t[:,trainIdx], ce, bias = bias)
    testData = dataSource(X[:,testIdx], t[:,testIdx], ce, bias = bias)
    return trainData, testData
  end
end

# ==========================================================================
# DataFrame labeled sources

immutable DataFrameLabeledDataSource <: LabeledDataSource
  dataFrame::DataFrame
  formula::Formula
end

# ==========================================================================
# Encode a DataFrameLabeledDataSource

function encodeDataSource{E<:ClassEncoding}(source::DataFrameLabeledDataSource, ::Type{E})
  mf = ModelFrame(source.formula, source.dataFrame)
  mm = ModelMatrix(mf)
  dfBias = in(0,mm.assign)
  X = dfBias ? mm.m[:,2:end]: mm.m
  extBias = dfBias * 1.0
  t = convert(Vector{Float64}, model_response(mf))
  ce = E(t)
  t_enc = labelencode(ce, t)
  dataSource(X', t_enc, ce, bias=extBias)
end

# ==========================================================================
# Choose best DataSource for the parameters

function dataSource{E<:ClassEncoding, N}(features::AbstractArray{Float64,2},
                                         targets::AbstractArray{Float64,N},
                                         encoding::E;
                                         bias::Float64 = 1.)
  EncodedInMemoryLabeledDataSource(features, targets, encoding, bias)
end

function dataSource(formula::Formula, data::DataFrame)
  DataFrameLabeledDataSource(data, formula)
end

function dataSource{E<:ClassEncoding}(formula::Formula, data::DataFrame, ::Type{E})
  encodeDataSource(DataFrameLabeledDataSource(data, formula), E)
end
