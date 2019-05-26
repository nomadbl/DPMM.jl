abstract type DPMMAlgorithm{P} end

_default_model(::Type{<:AbstractFloat}) = DPGMM
_default_model(::Type{<:Integer})       = DPDMM

function setup_workers(ncpu)
    if nworkers() != ncpu
        @warn("setting up parallel processes, takes a while for once!")
        addprocs(ncpu; exeflags="--project=$(dir())") # enable threaded blass
        @everywhere @eval Main using DPMM, SharedArrays, Distributed
        @info "workers: $(Main.Distributed.workers()) initialized"
    end
end

function initialize_clusters(X::AbstractMatrix, algo::DPMMAlgorithm{P}) where P
    labels    = random_labels(X,algo)
    clusters  = create_clusters(X,algo,labels)
    cluster0  = empty_cluster(algo)
    if P
        ws = workers()
        @info "sending model to workers"
        @everywhere ws (_model = $(algo.model))
        @info "sending data to workers"
        @everywhere ws (_X = $(X))
        @info "sending cluster0 to workers"
        @everywhere ws (_cluster0 = $(cluster0))
        return SharedArray(labels), clusters, cluster0
    end
    return labels, clusters, cluster0
end
