module MakieExt

using DPMM, Makie
"""

    setup_scene(X)

    Initialize plots for visualizing 2D data
"""
function DPMM.setup_scene(X)
    f, a, p = scatter(X[1,:],X[2,:],color=DPMM.colorpalette[ones(Int,size(X,2))],markersize=4)
    return (figure=f, axis=a, plot=p)
end

function DPMM.record!(scene::Any,z::AbstractArray,T::Int)
    z = first.(z)
    K=sort(unique(z))
    colors = map(zi->(findfirst(x->x==zi,K)-1)%12+1,z)
    scene.plot.color[] = DPMM.colorpalette[colors]
    scene.axis.title[] = "T=$T"
end
DPMM.record!(scene::Nothing,z::AbstractArray,T::Int) = nothing

export setup_scene

end
