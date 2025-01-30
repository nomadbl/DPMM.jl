module DPMM
import StatsAPI: fit
using Distributions, ColorBrewer, Colors, Distributed, SharedArrays, SparseArrays, LinearAlgebra, PDMats, Random #Makie

import Base: length, convert, size, *, +, -, isempty, getindex, sum, length, rand, @propagate_inbounds

const colorpalette  = RGBA.(palette("Set3", 12))

import SparseArrays: AbstractSparseMatrix, AbstractSparseVector, nonzeroinds, nonzeros


import Distributions: _rand!, partype, AbstractRNG, DirichletCanon,
                      _logpdf!, rand, pdf, params, _wishart_genA!, var,
                      mean, cov, params, invcov, logdetcov, sqmahal, sqmahal!,
                      partype, unwhiten_winv!,log2π, mvnormal_c0, _logpdf, lgamma,
                      xlogy, suffstats, SufficientStats, GenericMvTDist,
                      AliasTable

using MCMCDiagnosticTools, AbstractMCMC

import PDMats: unwhiten!, add!, quad, quad!

using TimerOutputs
const to = TimerOutput()

dir(path...) = joinpath(dirname(@__DIR__),path...)
"""
    AbstractCluster

Abstract type for clusters

Each subtype should provide the following methods:
- `population(c)`: population of the cluster
- `isempty(m::AbstractCluster)`: checks whether the cluster is empty?
- `logαpdf(c,x)` : log(∝likelihood) of a data point
- `lognαpdf(c,x)`: log(population) + logαpdf(c,x) for a data point (used in CRP calculations)
- `ClusterType(m::AbstractDPModel,X::AbstractArray)`  : constructor (X is the data as columns)
- `ClusterType(m::AbstractDPModel,s::SufficientStats)`: constructor

Other generic functions are implemented on top of these core functions.
"""
abstract type AbstractCluster end
const GenericClusters = Dict{Int, <:AbstractCluster}

"""
    population(m::AbstractCluster)

Number of data points in a cluster
"""
population(m::AbstractCluster)

"""
    logαpdf(m::AbstractCluster,x::AbstractArray)

log(∝likelihood) of a data point given by a cluster.
"""
logαpdf(m::AbstractCluster,x::AbstractArray)

@inline isempty(m::AbstractCluster)    = population(m)==0

"""
    lognαpdf(m::AbstractCluster,x::AbstractArray)

log(population) + log(∝likelihood) of a data point given by a cluster.
"""
@inline lognαpdf(m::AbstractCluster,x) = log(population(m)) + logαpdf(m,x)


include("Core/linearalgebra.jl")
include("Core/mvnormal.jl"); export MvNormalFast
include("Core/niw.jl"); export NormalWishart
include("Core/sparse.jl"); export DPSparseMatrix, DPSparseVector
include("Core/dirichletmultinomial.jl"); export DirichletFast
include("Core/algorithms.jl"); export run!, setup_workers, initialize_clusters
include("Core/chains.jl"); export DPMMChain
include("Data/data.jl");  export rand_with_label, RandMixture, GridMixture, RandDiscreteMixture
include("Data/nytimes.jl"); export readNYTimes
include("Data/visualize.jl"); export setup_scene
include("Models/model.jl")
include("Models/dpgmm.jl"); export DPGMM, DPGMMStats #, suffstats, updatestats, downdatestats, posterior, posterior_predictive
include("Models/dpmnmm.jl"); export DPMNMM, DPMNMMStats
include("Clusters/CollapsedCluster.jl"); export CollapsedCluster, CollapsedClusters
include("Clusters/DirectCluster.jl"); export DirectCluster, DirectClusters
include("Clusters/SplitMergeCluster.jl"); export SplitMergeCluster, SplitMergeClusters
include("Algorithms/CollapsedGibbs.jl"); export  CollapsedAlgorithm
include("Algorithms/DirectGibbs.jl"); export DirectAlgorithm
include("Algorithms/SplitMerge.jl"); export SplitMergeAlgorithm

"""
    fit(algorithm::Type{<:DPMMAlgorithm}, X::AbstractMatrix; ncpu=1, T=3000, benchmark=false, scene=nothing, o...)

`fit` is the main function of DPMM.jl which clusters given data matrix where columns are data points.

The output is the labels for each data point.

Default clustering algorithm is `SplitMergeAlgorithm`

Keywords:

- `ncpu=1` : the number of parallel workers.

- `T=3000` : iteration count

- `benchmarks=false` : if true returns elapsed time

- `scene=nothing`: plot scene for visualization. see [`setup_scene`](@ref)

- `chain::Union{Bool, DPMMChain}=false`: accumulate cluster parameter samples in DPMMChain.

- o... : other keyword argument specific to `algorithm`
"""
function fit(algorithm::Type{<:DPMMAlgorithm}, X::AbstractMatrix; ncpu=1, T=3000, benchmark::Val{B}=Val(false),
             scene=nothing, chain::Union{Bool, DPMMChain{ET, CT}}=false, o...) where {B, ET, CT}
    if ncpu>1
         setup_workers(ncpu)
    end
    algo = algorithm(X; parallel=ncpu>1, o...)
    labels, clusters, cluster0 = initialize_clusters(X,algo)
    mcmc_chain = if chain == true
        DPMMChain(typeof(cluster0))
    elseif chain isa DPMMChain
        @assert typeof(cluster0) == CT "DPMMChain cluster type $CT does not match algorithm cluster type $(typeof(cluster0))"
        chain
    else
        false
    end
    tres = @elapsed run!(algo, X, labels, clusters, cluster0; T=T, scene=scene, chain=mcmc_chain)
    @debug "$tres second passed"
    labels = first.(labels) # not return subclusters
    B && return labels, tres
    return labels
end

function chain_append!(chain::DPMMChain{T,CT}, samples::AbstractVector{AbstractVector{CT}}) where {T, CT<:AbstractCluster}
    nsamples, _, nchains = size(chain)
    @assert nchains == length(samples) "cannot append $(length(samples)) chains to DPMMChain containing $nchains chains"
    for chnind in eachindex(chain.value)
      push!(chain.value[chnind], samples[chnind])
    end
    itr = isempty(chain.iter) ? 1 : last(chain.iter)+1
    push!(chain.iter, itr)
end
chain_append!(chain::Bool,z) = nothing
chain_append!(chain::Nothing,z) = nothing

export fit

end # module

# include("Serial/CollapsedGibbs.jl"); export collapsed_gibbs
# include("Serial/QuasiCollapsedGibbs.jl");export quasi_collapsed_gibbs
# include("Serial/DirectGibbs.jl"); export direct_gibbs
# include("Serial/QuasiDirectGibbs.jl"); export quasi_direct_gibbs
# include("Serial/SplitMerge.jl"); export split_merge_gibbs, split_merge_gibbs!, split_merge_labels
# include("Parallel/DirectGibbsParallel.jl"); export direct_parallel!, direct_gibbs_parallel!
# include("Parallel/QuasiDirectParallel.jl"); export quasi_direct_parallel!,  quasi_direct_gibbs_parallel!
# include("Parallel/QuasiCollapsedParallel.jl");export quasi_collapsed_parallel!,  quasi_collapsed_gibbs_parallel!
# include("Parallel/SplitMergeParallel.jl");export splitmerge_parallel!, splitmerge_parallel_gibbs!
