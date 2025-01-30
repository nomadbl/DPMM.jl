module MakieExt

using DPMM, Makie, Distributions, LinearAlgebra

function getellipsepoints(cx, cy, rx, ry, θ)
	t = range(0, 2*pi, length=100)
	ellipse_x_r = @. rx * cos(t)
	ellipse_y_r = @. ry * sin(t)
	R = [cos(θ) sin(θ); -sin(θ) cos(θ)]
	r_ellipse = [ellipse_x_r ellipse_y_r] * R
	x = @. cx + r_ellipse[:,1]
	y = @. cy + r_ellipse[:,2]
	(x,y)
end

function getellipsepoints(μ, Σ, confidence=0.95)
	quant = Distributions.quantile(Chisq(2), confidence) |> sqrt
	cx = μ[1]
	cy =  μ[2]

	egvs = eigvals(Σ)
	if egvs[1] > egvs[2]
		idxmax = 1
		largestegv = egvs[1]
		smallesttegv = egvs[2]
	else
		idxmax = 2
		largestegv = egvs[2]
		smallesttegv = egvs[1]
	end

	rx = quant*sqrt(largestegv)
	ry = quant*sqrt(smallesttegv)

	eigvecmax = eigvecs(Σ)[:,idxmax]
	θ = atan(eigvecmax[2]/eigvecmax[1])
 	if θ < 0
		θ += 2*π
	end

	getellipsepoints(cx, cy, rx, ry, θ)
end
function DPMM.draw_gaussian_2d!(axis::Makie.Axis, μ, Σ, q::Real=0.95)
    lines!(axis, getellipsepoints(μ, Σ, q)..., label="$(q*100)% confidence interval")
end

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
