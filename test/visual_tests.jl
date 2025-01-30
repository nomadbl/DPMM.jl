using Distributions

function draw_clusters(scene, chain)
    clusters = chain.value[end][end]
    for cluster in clusters
        μ = Distributions.mean(posterior_predictive(cluster))
        Σ = Distributions.cov(posterior_predictive(cluster))
        DPMM.draw_gaussian_2d!(scene.axis, μ, Σ)
    end
end
@testset "visual tests" begin
    gmodel = GridMixture(2)
    X,labels = rand_with_label(gmodel,1000)
    scene = setup_scene(X)
    chain = DPMMChain(CollapsedAlgorithm, X)
    @info "Testing Collapsed Algorithm"
    labels = fit(CollapsedAlgorithm, X; T=1000, scene=scene, chain=chain)
    draw_clusters(scene, chain)
    display(scene.figure)
    @info "Testing Quasi-Collapsed Algorithm"
    labels = fit(CollapsedAlgorithm, X; quasi=true, T=1000, scene=scene)
    display(scene.figure)
    @info "Testing Direct Algorithm"
    labels = fit(DirectAlgorithm, X; T=1000, scene=scene)
    display(scene.figure)
    @info "Testing Quasi-Direct Algorithm"
    labels = fit(DirectAlgorithm, X; quasi=true, T=1000, scene=scene)
    display(scene.figure)
    @info "Testing Split Merge Algorithm"
    labels = fit(SplitMergeAlgorithm, X; T=1000, scene=scene)
    display(scene.figure)
    @info "Testing Split Merge Algorithm without Merge"
    labels = fit(SplitMergeAlgorithm, X; merge=true, T=1000, scene=scene)
    display(scene.figure)
    @test true
end
