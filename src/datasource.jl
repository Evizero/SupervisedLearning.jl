export DataSource, LabeledDataSource
export InMemoryLabeledDataSource
export nobs, nvar, features, targets
export dataSource

using ArrayViews
import StatsBase.nobs

abstract DataSource

nobs(source::DataSource) = throw(MethodError("Not implemented for the given datasource type"))
nvar(source::DataSource) = throw(MethodError("Not implemented for the given datasource type"))
features(source::DataSource) = throw(MethodError("Not implemented for the given datasource type"))
features(source::DataSource, offset::Int, length::Int) = throw(MethodError("Not implemented for the given datasource type"))

# labeled sources

abstract LabeledDataSource <: DataSource

targets(source::LabeledDataSource) = throw(MethodError("Not implemented for the given datasource type"))
targets(source::LabeledDataSource, offset::Int, length::Int) = throw(MethodError("Not implemented for the given datasource type"))

# In-memory labeled sources

immutable InMemoryLabeledDataSource{F<:FloatingPoint,N} <: LabeledDataSource
  features::AbstractArray{F,2}
  targets::AbstractArray{F,N}

  function InMemoryLabeledDataSource(features::AbstractArray{F,2}, targets::AbstractArray{F,1})
    size(features,2) == length(targets) || throw(DimensionMismatch("Features and targets have to have the same number of observations"))
    new(features, targets)
  end

  function InMemoryLabeledDataSource(features::AbstractArray{F,2}, targets::AbstractArray{F,2})
    size(features,2) == size(targets,2) || throw(DimensionMismatch("Features and targets have to have the same number of observations"))
    new(features, targets)
  end
end

function InMemoryLabeledDataSource{F<:FloatingPoint,N}(features::AbstractArray{F,2}, targets::AbstractArray{F,N})
   InMemoryLabeledDataSource{F,N}(features, targets)
end

nobs{F<:FloatingPoint,N}(source::InMemoryLabeledDataSource{F,N}) = size(source.features, 2)
nvar{F<:FloatingPoint,N}(source::InMemoryLabeledDataSource{F,N}) = size(source.features, 1)
features{F<:FloatingPoint,N}(source::InMemoryLabeledDataSource{F,N}) = source.features
features{F<:FloatingPoint,N}(source::InMemoryLabeledDataSource{F,N}, offset::Int, length::Int) = view(source.features, :, offset:(offset+length-1))
targets{F<:FloatingPoint,N}(source::InMemoryLabeledDataSource{F,N}) = source.targets
targets{F<:FloatingPoint}(source::InMemoryLabeledDataSource{F,1}, offset::Int, length::Int) = view(source.targets, offset:(offset+length-1))
targets{F<:FloatingPoint}(source::InMemoryLabeledDataSource{F,2}, offset::Int, length::Int) = view(source.targets, :, offset:(offset+length-1))

# Choose best DataSource for the parameters

function dataSource{F<:FloatingPoint,N}(features::AbstractArray{F,2}, targets::AbstractArray{F,N})
  InMemoryLabeledDataSource(features, targets)
end
