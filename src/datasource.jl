export DataSource, LabeledDataSource, InMemoryLabeledDataSource
export EncodedInMemoryLabeledDataSource
export nobs, nvar, features, targets, bias
export dataSource

using ArrayViews
using DataFrames
import StatsBase.nobs

abstract DataSource

nobs(source::DataSource) = throw(MethodError("Not implemented for the given datasource type"))
nvar(source::DataSource) = throw(MethodError("Not implemented for the given datasource type"))
features(source::DataSource) = throw(MethodError("Not implemented for the given datasource type"))
features(source::DataSource, offset::Int, length::Int) = throw(MethodError("Not implemented for the given datasource type"))

# labeled sources

abstract LabeledDataSource <: DataSource
abstract InMemoryLabeledDataSource <: LabeledDataSource

targets(source::LabeledDataSource) = throw(MethodError("Not implemented for the given datasource type"))
targets(source::LabeledDataSource, offset::Int, length::Int) = throw(MethodError("Not implemented for the given datasource type"))
bias(source::LabeledDataSource) = throw(MethodError("Not implemented for the given datasource type"))
nclasses(source::LabeledDataSource) = throw(MethodError("Not implemented for the given datasource type"))
labels(source::LabeledDataSource) = throw(MethodError("Not implemented for the given datasource type"))

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
    (typeof(encoding) <: OneOfKClassEncoding) && throw(MethodError("Can't have OneOutOfK-Encoding with a one vector as target"))
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

# ==========================================================================
# DataFrame labeled sources

immutable DataFrameLabeledDataSource <: LabeledDataSource
  dataFrame::DataFrame
  formula::Formula
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
  mf = ModelFrame(formula, data)
  mm = ModelMatrix(mf)
  dfBias = in(0,mm.assign)
  X = dfBias ? mm.m[:,2:end]: mm.m
  extBias = dfBias * 1.0
  t = convert(Vector{Float64}, model_response(mf))
  dataSource(X', t, bias = extBias)
end
