X = DPMM.readNYTimes(DPMM.dir("data/docword.nytimes.txt"))

@testset "nytimes" begin
    labels = fit(SplitMergeAlgorithm, X; ncpu=8, merge=false, T=5000)
    @show unique(labels)
    @test true
end
