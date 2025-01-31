@testset "serial algorithms" begin
    gmodel = GridMixture(2)
    X,labels = rand_with_label(gmodel,1000)
    @info "Testing Collapsed Algorithm"
    labels = fit(CollapsedAlgorithm, X; T=100)
    @info "Testing Quasi-Collapsed Algorithm"
    labels = fit(CollapsedAlgorithm, X; quasi=true, T=100)
    @info "Testing Direct Algorithm"
    labels = fit(DirectAlgorithm, X; T=100)
    @info "Testing Quasi-Direct Algorithm"
    labels = fit(DirectAlgorithm, X; quasi=true, T=100)
    @info "Testing Split Merge Algorithm"
    labels = fit(SplitMergeAlgorithm, X; T=100)
    @info "Testing Split Merge Algorithm without Merge"
    labels = fit(SplitMergeAlgorithm, X; merge=true, T=100)
    @test true
end
