using GLMakie, Makie
using DPMM, Images, GLMakie


@info "downloading water_coins.jpg"
img = load(download("http://docs.opencv.org/3.1.0/water_coins.jpg"));
img = Gray.(img)
display(img)
imghw = reshape(Float64.(channelview(img)), 1, :)
X = zeros(2,length(img))
inds = reshape(CartesianIndices(img),:)
X[1,:] .= map(i->i[2],inds)
X[2,:] .= map(i->i[1],inds)

scene = setup_scene(X)
@info "clustering water_coins.jpg"
labels = fit(SplitMergeAlgorithm, imghw; T=1000, α=5.0, scene=scene)
@info "displaying clustering"
display(scene.figure)

# @info "downloading flower.jpg"
# img = load(download("https://juliaimages.org/latest/assets/segmentation/flower.jpg"));
# display(Gray.(img))
# imghw = reshape(Float64.(channelview(img)), 3, :)
# X = zeros(2,length(img))
# inds = reshape(CartesianIndices(img),:)
# X[1,:] .= map(i->i[2],inds)
# X[2,:] .= map(i->i[1],inds)

# @info "clustering flower.jpg"
# scene = setup_scene(X)
# @info "displaying clustering"
# labels = fit(SplitMergeAlgorithm, imghw; T=100, α=5.0, scene=scene)
# display(scene.figure)
