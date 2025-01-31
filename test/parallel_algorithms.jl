@testset "parallel algorithms" begin
    gmodel = GridMixture(2)
    X,labels = rand_with_label(gmodel,1000)
    #labels = fit(CollapsedAlgorithm, X; T=100)
    @info "Testing Collapsed Algorithm with 2 worker"
    labels = fit(CollapsedAlgorithm, X; quasi=true, T=100, ncpu=3)
    @info "Testing Direct Algorithm with 2 worker"
    labels = fit(DirectAlgorithm, X; T=100, ncpu=3)
    @info "Testing Quasi-Direct Algorithm with 2 worker"
    labels = fit(DirectAlgorithm, X; quasi=true, T=100, ncpu=3)
    @info "Testing Split-Merge Algorithm with 2 workers"
    labels = fit(SplitMergeAlgorithm, X; T=100, ncpu=3)
    @info "Testing Split-Merge, without merge, Algorithm with 2 workers"
    labels = fit(SplitMergeAlgorithm, X; merge=true, T=100, ncpu=3)
    @test true
end
