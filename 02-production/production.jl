### A Pluto.jl notebook ###
# v0.12.18

using Markdown
using InteractiveUtils

# ╔═╡ d315f0e2-5754-11eb-2240-dd868564adfb
using Revise;

# ╔═╡ b63e580c-56b8-11eb-254c-b92db2e58e72
using ModelingToolkit;

# ╔═╡ 865062b4-56bf-11eb-13a4-292920055e10
"""
	module SolveMTK

Function to symbolically solve some equations defined by ModelingToolkit.
"""
module SolveMTK

using ModelingToolkit

vars(x::Sym) = Set([x])
vars(x::Number) = Set()
vars(t::Term) = union(vars.(ModelingToolkit.arguments(t))...)
vars(e::Equation) = union(vars(e.lhs), vars(e.rhs))

"""
	solve(eq, x; maxdepth=20)

Tries to solve the `eq` for `x`. 

# Arguments 
- `eq` should be a ModelingToolkit.Equation with fields `eq.lhs` and `eq.rhs`. 
- `x` should appear in `eq` and be a created by ModelingToolkit.@variables or @parameters

# Returns
An equation that should have `x` as the lhs variable and the solution as the rhs.

# Limitations
`eq` must only invole `+`, `-`, `*`, `/`, and `^` operations. Within non-commutative operations, `x` should only appear in one argument of the operation.

# Extending
To extend `solve` to work with additional operations, the `inverteq` must be defined for the additional operations. 
"""
function solve(eq, x; maxdepth=20)
	# ensure x is only on lhs
	if x ∈ vars(eq.rhs)
		eq = simplify(eq.lhs - eq.rhs ~ 0, polynorm=true)
	end
	
	i = 0
	while (eq.lhs isa ModelingToolkit.Term) && i <= maxdepth
	#for i in 1:1
		eq = move_op_right(eq, x)
		i += 1
	#end
	end
	eq = simplify(eq, polynorm=true)
	return(eq)	
end

function move_op_right(eq,x)
	op = ModelingToolkit.operation(eq.lhs)
	args = ModelingToolkit.arguments(eq.lhs)
	return(move_op_right(x, args, op, eq.rhs))
end

"""
	inverteq(op, xa, r, a, xfirst)

Converts an equation of the form `xa op a ~ r` to one with `xa ~ f(a, r)`. 

If `xfirst==false`, it assumes the original equation is `a op xa ~ r` instead.
"""
inverteq(op::typeof(+), xa, r, a, xfirst) = xa ~ r - a
inverteq(op::typeof(*), xa, r, a, xfirst) = xa ~ r/a
inverteq(op::typeof(-), xa, r, a, xfirst) = 
	xfirst ? x ~ r + a : x ~ a - r
inverteq(op::typeof(/), xa, r, a, xfirst) =
	xfirst ? (xa ~ r*a) : (xa ~ a/r)
inverteq(op::typeof(^), xa, r, a, xfirst) = 
	xfirst ? (xa ~ r^(1/a)) : (xa ~ log(r)/log(a))

function move_op_right(x, args, op::Commutes, rhs) where Commutes <: Union{typeof(+), typeof(*)}
	xin = (x .∈ vars.(args))	
	xa = reduce(op, args[xin])
	a = reduce(op, args[.!xin])
	eq = inverteq(op, xa, rhs, a, true)
	return(eq)
end

function move_op_right(x, args, op, rhs)
	length(args)==2 || error("Only 2 args allowed for noncommutative op")
	eq = if x ∈ vars(args[1]) && !(x ∈ vars(args[2]))
		inverteq(op, args[1], rhs, args[2], true)
	elseif x ∈ vars(args[2]) && !(x ∈ vars(args[1]))
		inverteq(op, args[2], rhs, args[1], false)
	else
		println("-----------------------------")
		println(args)
		println(op)
		error("x in wrong places")
	end
	return(eq)
end

end

# ╔═╡ 41769c24-567b-11eb-2f2d-9b51f92e7947
module Production

using Distributions, OffsetArrays

struct Params{T}
	βl::T
	βk::T
	ρ::T
	σϵ::T
	ση::T
end

function simulate(p::Params{R}, N, T; 
		wage=ones(N,T), price=ones(N,T), rent=ones(N,T)) where R
	y = Matrix{R}(undef, N, T)
	l = Matrix{R}(undef, N, T)
	k = Matrix{R}(undef, N, T)
	ϵ = rand(Normal(0, p.σϵ), N, T)
	η = rand(Normal(0, p.ση), N, T)
	ω = OffsetArray(Matrix{R}(undef, N, T+1), 1:N,0:T)
	ω[:,0] .= rand(Normal(0, p.ση/(1-p.ρ)), N)
	for t in 1:T		
		ω[:,t] .= ω[:,t-1]*p.ρ + η[:,t]
		# capital chosen at t-1 with rental price rent[i,t] and no adjustment cost 
		k[:,t] .= (1-p.βl)/(1-p.βk-p.βl)*(
			1/(1-p.βl)*log.(price[:,t]./(rent[:,t].*wage[:,t].^p.βl) 
					*mean(LogNormal(0,p.σϵ))) .+
			log(p.βl^(p.βl/(1-p.βl)) - p.βl^(1/(1-p.βl))) .+
			log(p.βk/(1-p.βl)) .+
			log.(mean.(LogNormal.(p.ρ*ω[:,t-1]/(1-p.βl), p.ση/(1-p.βl))))
			)
		
		l[:,t] .= 1/(1-p.βl)*(log(p.βl*mean(LogNormal(0,p.σϵ))) .+
				log.(price[:,t]./wage[:,t]) .+ p.βk*k[:,t] .+ ω[:,t])			
		
		y[:,t] .= p.βl*l[:,t] .+ p.βk*k[:,t] .+ ω[:,t] .+ ϵ[:,t]
	end	
	return((y=y,l=l,k=k,ω=ω))
end

end
		

# ╔═╡ 3c80dbde-568b-11eb-18ae-17ec1bb0cc92
#using AbstractPlotting, WGLMakie
using Plots, StatsPlots

# ╔═╡ b7e414a2-5687-11eb-1a72-073c31266c02
using DataFrames, GLM

# ╔═╡ c1c9492c-5689-11eb-2bed-2933c00e9c17
using FixedEffectModels

# ╔═╡ a6a60cb6-569d-11eb-22d8-61373838c6b2
html"<button onclick='present()'>present</button>"

# ╔═╡ fa80b794-569d-11eb-2f16-0dad9fa54602
md"""
# Production Functions
Paul Schrimpf

[UBC ECON567](https://faculty.arts.ubc.ca/pschrimpf/565/565.html) 

[![Creative Commons License](https://i.creativecommons.org/l/by-sa/4.0/88x31.png)](http://creativecommons.org/licenses/by-sa/4.0/)
[Creative Commons Attribution-ShareAlike 4.0 International License](http://creativecommons.org/licenses/by-sa/4.0/)
"""

# ╔═╡ 7d3e92a0-569d-11eb-3d2b-bdbfb6b4b364
md"""
In this notebook we will similate some realistic production function data, and use the simulated data to investigate the performance of various estimators.

## Model

The production function is Cobb-Douglas, so in logs it is:

```math
y_{it} = \beta_l \ell_{it} + \beta_k k_{it} + \omega_{it} + \epsilon_{it}
```

where $y$ is log output, $\ell$ is log labor, $k$ is log capital, $\omega$ is productivity, and $\epsilon$ is an ex-post shock.

Most production function estimation methods do not require fully specifying the data generating process for labor and capital. However, in this notebook, we want to simulate a complete dataset, so we need to fully specify how capital and labor are chosen. We will do so in a way that is consistent with economic theory and includes endogeneity problems, but is not too complicated.

### Labor

We assume that labor is chosen flexibly and output and input markets are competitive. That is, labor is chosen when the firm knows $\omega_{it}$, and there are no adjustment frictions in labor. Firms choose labor to maximize expected profits:

```math
\max_L E[ p_{it} e^{\omega_{it} + \epsilon_{it}} L^{\beta_{\ell}} K_{it}^{\beta_k} - w_{it} L | \omega_{it}, K_{it}, w_{it}, p_{it} ]
```

Solving for L, 
```math
L(K,\omega,p,w) = \left( E[e^\epsilon] \frac{p}{w} \beta_\ell e^\omega \right)^{1/(1-\beta_\ell)} K^{\beta_k/(1-\beta_\ell)}
```
or with labor and capital in logs, 
```math
\ell(k, \omega, p, w) = \frac{1}{1-\beta_\ell} \log \left( E[e^\epsilon] \frac{p}{w} \beta_\ell \right) + \frac{1}{1-\beta_\ell} \omega + \frac{\beta_k}{1-\beta_\ell} k
```
"""

# ╔═╡ 2dd656c6-56d2-11eb-1e1d-d3636d80d71a
md"""
### Digression: Using Julia as a CAS

We can check our calculation of labor using a computer algebra system (CAS). Probably the most popular CAS is Mathematica / Wolfram Alpha. We can accomplish something similar in Julia using the [ModelingToolkit.jl](https://github.com/SciML/ModelingToolkit.jl) package. The focus of this package is on conveniently converting from the sort of symbolic math that we would write in papers to fast numeric code that can be used with numeric solvers. Consequently, the package does not yet have convenient interface for solving equations symbolically.

However, a small extension allows it to solve some simple equations symbolically. The `module SolveMTK` contains functions that try to solve equations symbolically. Don't worry too much about the code inside `module SolveMTK`; it is fairly advanced.
"""

# ╔═╡ 5cc072ac-56d3-11eb-18b8-7f3238cdbec7
md"""
The code cell below uses ModelingToolkit to symbolically define the firm's profit function. It then symbolically computes the first order condition, and solves for L. Hopefully the output is the same as what is written above.
"""

# ╔═╡ 68f6e53e-56d4-11eb-3575-05ee3be38745
md"""
### Capital

We will assume that $K_{it}$ is chosen in advance at time $t-1$. There are no adjustment costs or other frictions in the choice of capital. We will assume that the firm rents capital at rental rate $r$. If prices ($p$, $w$, and $r$) evolve over time, we assume that firms have perfect foresight over them. Thus, the only uncertainty at time $t-1$ about time $t$ is about $\epsilon_{it}$ and $\omega_{it}$.

We also assume that $\omega$, follows an AR(1) process:
```math
\omega_{it} = \rho \omega_{it-1} + \eta_{it}
```

The firm's choice of capital maximizes expected profits:
```math
\max_{K} E\left[p \exp(\omega_{it} + \epsilon_{it}) L(K,\omega_{it}, p, w)^{\beta_\ell}  K^{\beta_k} - rK - wL(K,\omega_{it}, p, w)| \omega_{it-1}, p, w, r \right].
```
Substituting in for $L$, 
```math
\max_{K} E\left[p e^{\omega_{it} + \epsilon_{it}} \left( E[e^\epsilon] \frac{p}{w} \beta_\ell e^{\omega_it} \right)^{\beta_\ell/(1-\beta_\ell)} K^{\beta_k/(1-\beta_\ell)} - rK - w \left( E[e^\epsilon] \frac{p}{w} \beta_\ell e^{\omega_{it}} \right)^{1/(1-\beta_\ell)} K^{\beta_k/(1-\beta_\ell)} | \omega_{it-1}, p, w, r \right]
```
and simplifying,
```math
\max_{K} E\left[ \left(e^{\omega_{it}} E[e^\epsilon] \frac{p}{w^{\beta_\ell}} \right)^{1/(1-\beta_\ell)} \left(\beta_\ell^{\beta_\ell/(1-\beta_\ell)} - \beta_\ell^{1/(1-\beta_\ell)} \right)  K^{\beta_k/(1-\beta_\ell)} - rK | \omega_{it-1}, p, w, r \right]
```


taking the first order condition, and solving for $K$ gives:
```math
K(\omega_{it-1}, p, w, r) = 
\left( \frac{p}{r w^{\beta_\ell}} E[e^\epsilon] \right)^{1/(1-\beta_k-\beta_\ell)}
\left[\frac{\beta_k}{1-\beta_\ell}\left( \beta_\ell^{\beta_\ell/(1-\beta_\ell)} - \beta_\ell^{1/(1-\beta_\ell)}\right) \right]^{(1-\beta_\ell)/(1-\beta_k - \beta_\ell)} 
E[e^{\omega_{it} \frac{1}{1- \beta_k - \beta_\ell}} | \omega_{it-1}]
```

Note that we need to have $\beta_k + \beta_\ell < 1$ for this to be the correct solution. Without any adjustment frictions, decreasing returns to scale are required for the optimal choice of capital to be finite.
"""


# ╔═╡ 4be9eee0-56dd-11eb-3458-e3ad0d5c1a78
md"""

# Simulation

The code below simulates the model. We assume that $\epsilon$ and $\eta$ are normally distributed, so that the expectations of $e^\epsilon$ and $e^\omega$ are easy to compute.

"""

# ╔═╡ 8d1cfa5c-56a7-11eb-339e-3d733e6041ce
N, T = 200, 10

# ╔═╡ a16f916a-5687-11eb-0219-7b16fa743194
p = Production.Params(0.3, 0.6, 0.8, 1.0, 0.2)

# ╔═╡ b5eeabfe-56dd-11eb-0292-43f1e2166311
md""" 
Having more variation in wages, capital rental rates, and output prices will improve the performance of most estimators. 

If there is no variation in prices, $k_{it+1}$ and $\ell_{it}$ are perfectly collinear (they are both equal to affine functions of just $\eta_{it}$). This leads to the identification problems discussed by ACF and GNR. 

> Pluto notebooks are reactive: modifying wage or rent in the next two cells will cause all other cells that depend on them to be re-run. Try modifying them to see how the graphs and estimates below change.
"""

# ╔═╡ da42b1c0-56b4-11eb-2a47-531922e9e6d4
#wage = ones(N,T); 
wage = rand(N,T) .+ 0.5;

# ╔═╡ d7d9043e-56b4-11eb-05af-1f838362a37d
#rent = ones(N,T); 
rent = rand(N,T) .+ 0.5;

# ╔═╡ b821a380-5687-11eb-0f31-89930c22a2ff
y, l, k, ω = Production.simulate(p, N, T, wage=wage, rent=rent);

# ╔═╡ 3eaaf80e-56b8-11eb-1f22-a380fc7122e9
let 
	@parameters βl, βk, ω, Ee, p, w # Ee is E[exp(ϵ)], others should be 
	 								# self-explanatory
	@variables L, K
	@derivatives Dl'~L
	profits = p*Ee*exp(ω)*L^βl*K^βk - w*L
	focL = ModelingToolkit.expand_derivatives(Dl(profits)) ~ 0
	SolveMTK.solve(focL,L)	
end

# ╔═╡ 0482c232-56ad-11eb-3e62-d5482fd641e1
Plots.gr(fmt="png") # need a bitmap format when plotting thousands of points or your browser will not be happy

# ╔═╡ 56fef692-568d-11eb-16c3-f7d64808f6d9
begin
	f=corrplot([vec(k)  vec(l)  vec(y)], #, resolution=(500,500))
			   label=["capital" "labor" "output"])
	f
end

# ╔═╡ b6cdfe5a-56de-11eb-2a6f-6146c7c3b1a4
md"""
# Estimates

## OLS

OLS is biased since $\ell_{it}$ and $k_{it}$ are correlated with $\omega_{it}$. Nonetheless, here are the estimates. If nothing else, it illustrates how to compute OLS with Julia.
"""

# ╔═╡ 1d0166b8-5751-11eb-26d7-d73619cec13e
import Dialysis.panellag

# ╔═╡ 62d3baf6-5689-11eb-04dd-19ce30089417
df = let 
	N,T=size(y)
	id = (1:N)*ones(Int,T)'
	t = ones(Int,N)*(1:T)'
	df=DataFrame(id=vec(id), t=vec(t), y = vec(y), l=vec(l), k=vec(k), 
				 ω=vec(ω[:,1:T]), w=vec(wage), r=vec(rent))
	sort!(df, [:id, :t])
	df[!,:klag] = panellag(:k, df, :id, :t, 1)
	df[!,:invest] = df[!,:k] - df[!,:klag]
	df	
end;

# ╔═╡ cfe6dbc8-5689-11eb-2c02-573501965578
lm(@formula(y ~ k + l), df)

# ╔═╡ c2df2432-56e1-11eb-1074-25f2d665cfed
md"""
### Infeasible OLS

For comparison, we can compute an "infeasible" OLS estimate where $\omega$ is observed.
"""

# ╔═╡ e994ccb2-56e1-11eb-3bc3-bd3ecb96aff6
lm(@formula(y ~ k + l + ω), df)

# ╔═╡ e2f499a8-56de-11eb-10af-9f1da128329f
md"""
## Fixed Effects

Fixed effects are also biased for this data generating process.

Note: I find that the `reg` function from the `FixedEffectModels` package has a more convenient way to compute robust and/or clustered standard errors than `lm` from `GLM`. It may be worth it to use `reg` for any IV or OLS regression, even without fixed effects. 
"""

# ╔═╡ bb829c62-5689-11eb-2bf7-6fdbde62cd01
reg(df, @formula(y ~ k + l + fe(id) + fe(t)), Vcov.cluster(:id, :t))

# ╔═╡ 3c26beb0-56e0-11eb-0973-ff465e329271
md"""
## IV

Since this data is simulated with firms taking prices as given, if there is variation in wages and rental rates, then they can be used as instruments. 

"""

# ╔═╡ 780a441a-56e0-11eb-1b86-3b988c8c1d07
reg(df, @formula(y ~ (k + l ~ r + w) + t), Vcov.cluster(:id, :t))
# including t in the formula shouldn't be needed, but there's an error when there's no exogenous variables...

# ╔═╡ 1979e336-56e0-11eb-3f7e-dfb33e4ca1fe
md"""
## Dynamic Panel
"""

# ╔═╡ 21c66388-56e0-11eb-2e3c-618a1d2c0362
md"""
## Olley-Pakes
"""

# ╔═╡ 6a2d413c-574c-11eb-39b5-cfca30ea3acc
module Estimators

import Dialysis 

function olleypakes(df, output, flexibleinputs, fixedinputs, controlvars, id, t;
	step1degree=4, step2degree=4)
	# step 1
	step1, eyex = Dialysis.partiallinear(output, flexibleinputs, controlvars, df, 
		npregress=(xp, xd, yd)->Dialysis.polyreg(xp,xd,yd, degree=step1degree), 
		clustervar=id)
	βflex = step1.coef[2:end]
	Vflex = step1.vcov[2:end,2:end]
	
	# step 2
	dc=copy(df[!,[output, flexibleinputs..., fixedinputs..., controlvars..., id, t]])
	dc[!,:ỹ] = dc[!,output] - Matrix(dc[!,flexibleinputs])*βflex	
	dc[!,:f] = eyex[:,1]
	function nlobj(β; dc=dc)
		obj = zero(eltype(β))
		for r in eachrow(dc) 
			
		dc[!,:ω̂] = dc[!,:f] .- Matrix(dc[!,:fixedinputs])*βk
		dc[!,:
		g = Dialysis.polyreg(
		return nothing	
	end
	
	return(step1, eyex)	
end

end

# ╔═╡ 2ae11fa8-5797-11eb-0232-f3e685dfc51c


# ╔═╡ f15e4ee8-5757-11eb-1485-25a2bad8ac3d


# ╔═╡ ef937930-5750-11eb-2511-5d7d033d6854
step1, eyex=Estimators.olleypakes(df, :y, [:l], [:k], [:klag, :invest], :id, :t)

# ╔═╡ bfc0c600-5754-11eb-2560-9701e3c85f02
eyex

# ╔═╡ Cell order:
# ╟─a6a60cb6-569d-11eb-22d8-61373838c6b2
# ╟─fa80b794-569d-11eb-2f16-0dad9fa54602
# ╠═d315f0e2-5754-11eb-2240-dd868564adfb
# ╟─7d3e92a0-569d-11eb-3d2b-bdbfb6b4b364
# ╟─2dd656c6-56d2-11eb-1e1d-d3636d80d71a
# ╠═b63e580c-56b8-11eb-254c-b92db2e58e72
# ╟─865062b4-56bf-11eb-13a4-292920055e10
# ╟─5cc072ac-56d3-11eb-18b8-7f3238cdbec7
# ╠═3eaaf80e-56b8-11eb-1f22-a380fc7122e9
# ╟─68f6e53e-56d4-11eb-3575-05ee3be38745
# ╟─4be9eee0-56dd-11eb-3458-e3ad0d5c1a78
# ╠═41769c24-567b-11eb-2f2d-9b51f92e7947
# ╠═8d1cfa5c-56a7-11eb-339e-3d733e6041ce
# ╠═a16f916a-5687-11eb-0219-7b16fa743194
# ╟─b5eeabfe-56dd-11eb-0292-43f1e2166311
# ╠═da42b1c0-56b4-11eb-2a47-531922e9e6d4
# ╠═d7d9043e-56b4-11eb-05af-1f838362a37d
# ╠═b821a380-5687-11eb-0f31-89930c22a2ff
# ╠═3c80dbde-568b-11eb-18ae-17ec1bb0cc92
# ╠═0482c232-56ad-11eb-3e62-d5482fd641e1
# ╠═56fef692-568d-11eb-16c3-f7d64808f6d9
# ╟─b6cdfe5a-56de-11eb-2a6f-6146c7c3b1a4
# ╠═b7e414a2-5687-11eb-1a72-073c31266c02
# ╠═1d0166b8-5751-11eb-26d7-d73619cec13e
# ╠═62d3baf6-5689-11eb-04dd-19ce30089417
# ╠═cfe6dbc8-5689-11eb-2c02-573501965578
# ╟─c2df2432-56e1-11eb-1074-25f2d665cfed
# ╠═e994ccb2-56e1-11eb-3bc3-bd3ecb96aff6
# ╟─e2f499a8-56de-11eb-10af-9f1da128329f
# ╠═c1c9492c-5689-11eb-2bed-2933c00e9c17
# ╠═bb829c62-5689-11eb-2bf7-6fdbde62cd01
# ╟─3c26beb0-56e0-11eb-0973-ff465e329271
# ╠═780a441a-56e0-11eb-1b86-3b988c8c1d07
# ╠═1979e336-56e0-11eb-3f7e-dfb33e4ca1fe
# ╠═21c66388-56e0-11eb-2e3c-618a1d2c0362
# ╠═6a2d413c-574c-11eb-39b5-cfca30ea3acc
# ╠═2ae11fa8-5797-11eb-0232-f3e685dfc51c
# ╠═f15e4ee8-5757-11eb-1485-25a2bad8ac3d
# ╠═ef937930-5750-11eb-2511-5d7d033d6854
# ╠═bfc0c600-5754-11eb-2560-9701e3c85f02
