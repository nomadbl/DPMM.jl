"""
    DPMMChain
MCMC Chain type for storing sampled clusters during the MCMC simulation, representing
samples x vals x chains, but allowing for varying number of parameters along the chain to
account for varying number of clusters during DPMM sampling.


`DPMMChain(algorithm::Type{<:DPMMAlgorithm}, X::AbstractMatrix; o...)`


Examples:

`DPMMChain(SplitMergeAlgorithm, rand(2, 100); model_type=DPMM.DPGMM)`
`DPMMChain(SplitMergeAlgorithm, rand(10, 100); model_type=DPMM.DPMNMM)`
"""
struct DPMMChain{CT<:AbstractCluster, IT<:AbstractVector{<:Integer}, A<:AbstractVector{<:AbstractVector{<:AbstractVector{CT}}},K<:NamedTuple,I<:NamedTuple} <: AbstractMCMC.AbstractChains
    value::A # Vector{Vector{Vector{clusters}}} <=> chains[samples[clusters[]]]
    iters::IT
    name_map::K
    info::I
end
emptychain(t::Type{<:AbstractCluster}) = Vector{Vector{t}}[Vector{Vector{t}}(),]
DPMMChain(t::Type{<:AbstractCluster}) = DPMMChain{t, Vector{Int64}, Vector{Vector{Vector{t}}}, NamedTuple{}, NamedTuple{}}(emptychain(t), Int64[], NamedTuple(), NamedTuple())
function DPMMChain(algorithm::Type{<:DPMMAlgorithm}, X::AbstractMatrix; o...)
    algo = algorithm(X; o...)
    labels, clusters, cluster0 = initialize_clusters(X,algo)
    DPMMChain(eltype(values(clusters)))
end
"""
    names(chains::DPMMChain)

Return the parameter names in the `chains`.
"""
Base.names(chains::DPMMChain{CT}) where {CT<:AbstractCluster} = fieldnames(CT)

"""
    names(chains::Chains, section::Symbol)

Return the parameter names of a `section` in the `chains`.
"""
Base.names(chains::DPMMChain, section::Symbol) = convert(Vector{Symbol}, chains.name_map[section])

"""
    names(chains::Chains, sections)

Return the parameter names of the `sections` in the `chains`.
"""
function Base.names(c::DPMMChain, sections)
    names = Symbol[]
    for section in sections
        append!(names, c.name_map[section])
    end
    return names
end
Base.keys(c::DPMMChain) = names(c)
Base.size(c::DPMMChain) = (length(c.iters), missing, length(c.value))
Base.size(c::DPMMChain, ind) = size(c)[ind]
Base.length(c::DPMMChain) = size(c, 1) * size(c,3)
Base.first(c::DPMMChain) = first(c.iters)
Base.last(c::DPMMChain) = last(c.iters)
Base.step(c::DPMMChain) = last(c)
function AbstractMCMC.chainscat(c::DPMMChain{CT}, cs::DPMMChain{CT}...; dims=3) where {CT<:AbstractCluster}
    # add a chain
    if dims == 3
        _cat(Val(3), c, cs...)
    # add samples to chains
    elseif dims == 1
        _cat(Val(1), c, cs...)
    # adding clusters to existing samples is undefined
    else
        throw(ArgumentError("chainscat: got dims==2, DPMMChain cannot be concatenated along parameter dimensions."))
    end
end
function _cat(::Val{1}, c1::DPMMChain{CT}, cs::DPMMChain{CT}...) where {CT<:AbstractCluster}
    # check inputs
    lastiter = last(c1)
    for c in cs
        first(c) > lastiter || throw(ArgumentError("iterations have to be sorted"))
        lastiter = last(c)
    end
    nms = names(c1)
    all(c -> names(c) == nms, cs) || throw(ArgumentError("chain names differ"))
    nchains = length(c1.value)
    all(c -> length(c.value) == nchains, cs) || throw(ArgumentError("number of chains differ"))
    chain_length = sum(length, c1.value)
    all(c -> sum(length, c.value) == chain_length, cs) || throw(ArgumentError("chain lengths differ"))
    nfo = c1.info
    all(c -> c.info == nfo, cs) || throw(ArgumentError("chain infos differ"))

    # concatenate all chains to respective chains in cs
    iters = vcat(c1.iters, [c.iter for c in cs]...)
    value = [vcat(c1.value[chnindex], [c.value[chnindex] for c in cs]...) for chnindex in eachindex(c1.value)]
    return DPMMChain(value, iters, c1.name_map, c1.info)
end
function _cat(::Val{3}, c1::DPMMChain{CT}, cs::DPMMChain{CT}...) where {CT<:AbstractCluster}
    # check inputs
    rng = c1.iters
    all(c -> c.iters == rng, cs) || throw(ArgumentError("chain sample indices differ"))
    nms = names(c1)
    all(c -> names(c) == nms, cs) || throw(ArgumentError("chain names differ"))
    nmp = c1.name_map
    all(c -> c.name_map == nmp, cs) || throw(ArgumentError("chain name maps differ"))
    nfo = c1.info
    all(c -> c.info == nfo, cs) || throw(ArgumentError("chain infos differ"))

    # # combine names and sections of parameters
    # nms = names(c1)
    # n = length(nms)
    # for c in cs
    #     nms = union(nms, names(c))
    #     n += length(names(c))
    #     n == length(nms) || throw(ArgumentError("non-unique parameter names"))
    # end

    # name_map = mapreduce(c -> c.name_map, merge_union, args; init = c1.name_map)

    # concatenate new chains
    value = vcat(c1.value, [c.value for c in cs]...)
    return DPMMChain(value, c1.iters, c1.name_map, c1.info)
end

"""
    var_of_information(chains::DPMMChain, x::AbstractMatrix; start_iter=1)


Mean and variance for Variation of information calculated between all pairs of clusterings in each chain, calculated on the dataset `x`.

`x` should be `D` by `N` for `D` dimensions and `N` data points.
"""
function var_of_information(chains::DPMMChain, x::AbstractMatrix; start_iter=1)
    nsamples, _, nchains = size(chains)
    vi = similar(x, nsamples*(nsamples-1)/2, nchains) # length of upper triangular matrix
    for c in 1:nchains,
        vi_ind = 1
        for i in eachindex(chains.value),
            j in eachindex(chains.value)
            if i > i
                vi[vi_ind, c] = var_of_information(chains.value[c][i], chains.value[c][j])
                vi_ind += 1
            end
        end
    end
    return vi
end
function R_hat_diagnostic(vi::AbstractMatrix)
    n, m = size(vi)
    # calculate mean and var of vi in each chain
    Ψj = mean(vi; dims=1)
    Ψ = mean(ψj)
    # between chain variance
    b = n / (m-1) * sum(abs2, ψj - ψ)
    # within chain variance
    w = 1 / (n-1) * mean(sum(abs2, vi - ψj; dims=1))
    # variance estimate
    v = (n - 1)/n * w + b / n
    # R̂ diagnostic
    R = sqrt(v/w)
    return R
end
# TODO: show function
