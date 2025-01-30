
@testset "visual tests" begin
    gmodel = GridMixture(2)
    X,labels = rand_with_label(gmodel,1000)
    scene = setup_scene(X)
    @info "Testing Collapsed Algorithm"
    labels = fit(CollapsedAlgorithm, X; T=1000, scene=scene)
    @info "Testing Quasi-Collapsed Algorithm"
    labels = fit(CollapsedAlgorithm, X; quasi=true, T=1000, scene=scene)
    @info "Testing Direct Algorithm"
    labels = fit(DirectAlgorithm, X; T=1000, scene=scene)
    @info "Testing Quasi-Direct Algorithm"
    labels = fit(DirectAlgorithm, X; quasi=true, T=1000, scene=scene)
    @info "Testing Split Merge Algorithm"
    labels = fit(SplitMergeAlgorithm, X; T=1000, scene=scene)
    @info "Testing Split Merge Algorithm without Merge"
    labels = fit(SplitMergeAlgorithm, X; merge=true, T=1000, scene=scene)
    @test true
end
