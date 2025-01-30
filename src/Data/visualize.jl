
# """

#     setup_scene(X)

#     Initialize plots for visualizing 2D data
# """
# function setup_scene(X)
#     if !isdefined(Main,:scene)
#         # @warn "setting up the plot, takes a while for once"
#         # @eval Main using CairoMakie
#     end
#     @eval Main scene = scatter($(X[1,:]),$(X[2,:]),color=DPMM.colorpalette[ones(Int,$(size(X,2)))],markersize=0.1)
#     @eval Main display(scene)
#     return Main.scene
# end
#  this code was moved to extension MakieExt
function draw_gaussian_2d! end
setup_scene(x::Nothing) = nothing
record!(scene::Nothing, o...) = nothing
