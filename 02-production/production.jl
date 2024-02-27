### A Pluto.jl notebook ###
# v0.19.19

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
end

# ╔═╡ d315f0e2-5754-11eb-2240-dd868564adfb
begin
  import Pkg
  Pkg.activate(Base.current_project())
  #Pkg.instantiate()
end


# ╔═╡ 2a662708-d5f0-41be-bc2e-56e00e2c975d
using Revise

# ╔═╡ 41769c24-567b-11eb-2f2d-9b51f92e7947
module Production

using Distributions, OffsetArrays

struct Params{T}
	βl::T
	βk::T
	ρ::T
	σϵ::T
	σξ::T
end

function simulate(p::Params{R}, N, T;
		wage=ones(N,T), price=ones(N,T), rent=ones(N,T),
		pricesknown=true, inertia = 0.0) where R
	y = Matrix{R}(undef, N, T)
	l = Matrix{R}(undef, N, T)
	k = Matrix{R}(undef, N, T)
	ϵ = rand(Normal(0, p.σϵ), N, T)
	ξ = rand(Normal(0, p.σξ), N, T)
	ω = OffsetArray(Matrix{R}(undef, N, T+1), 1:N,0:T)
	ω[:,0] .= rand(Normal(0, p.σξ/(1-p.ρ)), N)
	for t in 1:T
		ω[:,t] .= ω[:,t-1]*p.ρ + ξ[:,t]
		# capital chosen at t-1 with rental price rent[i,t] and no adjustment cost
		kopt = (1-p.βl)/(1-p.βk-p.βl)*(
			1/(1-p.βl)*
			(pricesknown ?  # prices known in advance,
				log.(price[:,t]./(rent[:,t].*wage[:,t].^p.βl)
					*mean(LogNormal(0,p.σϵ)))
				: # prices not known in advance, so take the mean
				log.(mean(price./(rent.*wage.^p.βl))
							*mean(LogNormal(0,p.σϵ)))
			) .+
			log(p.βl^(p.βl/(1-p.βl)) - p.βl^(1/(1-p.βl))) .+
			log(p.βk/(1-p.βl)) .+
			log.(mean.(LogNormal.(p.ρ*ω[:,t-1]/(1-p.βl), p.σξ/(1-p.βl))))
			)
		if t==1
			k[:,t] .= kopt
		else
			k[:,t] .= (1-inertia)*kopt .+ inertia*k[:,t-1]
		end

		l[:,t] .= 1/(1-p.βl)*(log(p.βl*mean(LogNormal(0,p.σϵ))) .+
				log.(price[:,t]./wage[:,t]) .+ p.βk*k[:,t] .+ ω[:,t])

		y[:,t] .= p.βl*l[:,t] .+ p.βk*k[:,t] .+ ω[:,t] .+ ϵ[:,t]
	end
	return((y=y,l=l,k=k,ω=ω))
end

end


# ╔═╡ 0437683c-5837-11eb-1f1c-33755302a653
using PlutoUI

# ╔═╡ 3c80dbde-568b-11eb-18ae-17ec1bb0cc92
using Plots, StatsPlots

# ╔═╡ b7e414a2-5687-11eb-1a72-073c31266c02
using DataFrames, GLM

# ╔═╡ c1c9492c-5689-11eb-2bed-2933c00e9c17
using FixedEffectModels

# ╔═╡ fa80b794-569d-11eb-2f16-0dad9fa54602
md"""
# Production Functions
Paul Schrimpf

[UBC ECON567](https://faculty.arts.ubc.ca/pschrimpf/565/565.html)

[![Creative Commons License](https://i.creativecommons.org/l/by-sa/4.0/88x31.png)](http://creativecommons.org/licenses/by-sa/4.0/)
[Creative Commons Attribution-ShareAlike 4.0 International License](http://creativecommons.org/licenses/by-sa/4.0/)
"""

# ╔═╡ dc247c6f-52fb-47ef-bd54-2c5a20cfe791
x = 2

# ╔═╡ 8a4ade3b-eb29-4762-b32e-34adb96c9773
z = 2

# ╔═╡ 16bfc249-ff90-45ea-a707-e96d59b30d84
x+z

# ╔═╡ 3ad05053-18cd-4e0a-92b3-0ced08d9a1f0
begin
	a = 1
	b = 2
	a + b 
end

# ╔═╡ 1641e5b8-2a1f-4a37-85f1-84c4bab78932
PlutoUI.TableOfContents()

# ╔═╡ 7d3e92a0-569d-11eb-3d2b-bdbfb6b4b364
md"""
In this notebook we will similate some production function data, and use the simulated data to investigate the performance of various estimators.

This notebook is meant to accompany [these slides](https://faculty.arts.ubc.ca/pschrimpf/565/02-1-production-methods.pdf), and the notation used should be similar.

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

# ╔═╡ 68f6e53e-56d4-11eb-3575-05ee3be38745
md"""
### Capital

We will assume that $K_{it}$ is chosen in advance at time $t-1$. There are no adjustment costs or other frictions in the choice of capital. We will assume that the firm rents capital at rental rate $r$. Firms can either know prices perfectly, or they assume prices evolve independently of $\epsilon$ and $\omega$.

We also assume that $\omega$, follows an AR(1) process:
```math
\omega_{it} = \rho \omega_{it-1} + \xi_{it}
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
K^\ast(\omega_{it-1}, p, w, r) =
\left( \frac{p}{r w^{\beta_\ell}} E[e^\epsilon] \right)^{1/(1-\beta_k-\beta_\ell)}
\left[\frac{\beta_k}{1-\beta_\ell}\left( \beta_\ell^{\beta_\ell/(1-\beta_\ell)} - \beta_\ell^{1/(1-\beta_\ell)}\right) \right]^{(1-\beta_\ell)/(1-\beta_k - \beta_\ell)}
E[e^{\omega_{it} \frac{1}{1- \beta_k - \beta_\ell}} | \omega_{it-1}]
```

Note that we need to have $\beta_k + \beta_\ell < 1$ for this to be the correct solution. Without any adjustment frictions, decreasing returns to scale are required for the optimal choice of capital to be finite.


For some combinations of estimators and other simulation settings, the frictionless choice of $K^\ast$ will cause identification problems. Therefore, we also consider a value of $k_t$ with inertia given by:
```math
k_{it} = (1 - \mathtt{inertia}) \log K^\ast + \mathtt{inertia} k_{it-1}
```
"""


# ╔═╡ 4be9eee0-56dd-11eb-3458-e3ad0d5c1a78
md"""

# Simulation

The code below simulates the model. We assume that $\epsilon$ and $\eta$ are normally distributed, so that the expectations of $e^\epsilon$ and $e^\omega$ are easy to compute.

"""

# ╔═╡ 55bc0887-d01d-4174-bb77-45e18989bffc


# ╔═╡ 90a815d7-f0fd-455c-a0b0-42eee3d91f6a


# ╔═╡ 8d1cfa5c-56a7-11eb-339e-3d733e6041ce
N, T = 1000, 10

# ╔═╡ a16f916a-5687-11eb-0219-7b16fa743194
p = Production.Params(0.6,  0.3, 0.5, 0.1, 0.1)

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

# ╔═╡ 2d7f4818-5837-11eb-3679-dfaad86cd792
md"""
> We can also create html elements that modify Julia variables.

Inertia: 0
$(@bind inertia Slider(range(0.,1., step=0.01))) 1

Prices known: $(@bind pricesknown CheckBox())
"""

# ╔═╡ b821a380-5687-11eb-0f31-89930c22a2ff
y, l, k, ω = Production.simulate(p, N, T, wage=wage, rent=rent,
								 pricesknown=pricesknown, inertia = inertia)

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
	df[!,:invest] = df[!,:k] - panellag(:k, df, :id, :t, -1)
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

# ╔═╡ 21c66388-56e0-11eb-2e3c-618a1d2c0362
md"""
## Olley-Pakes
"""

# ╔═╡ 6a2d413c-574c-11eb-39b5-cfca30ea3acc
module Estimators

import Dialysis
import Dialysis.panellag
import Optim

nomissing(x::Array{Union{Missing, T},D}) where T where D= Array{T,D}(x)
nomissing(x::Array{T}) where T <: Number = x

function olleypakes(df, output, flexibleinputs, fixedinputs, controlvars, id, t;
	step1degree=4, step2degree=4)
	df = sort!(df, [id, t])
	# step 1
	step1, eyex = Dialysis.partiallinear(output, flexibleinputs, controlvars, df,
		npregress=(xp, xd, yd)->Dialysis.polyreg(xp,xd,yd, degree=step1degree),
		clustervar=id)
	βl = step1.coef[2:end]
	Vl = step1.vcov[2:end,2:end]
	f̂ = eyex[:,1] - eyex[:,2:end]*βl
		#df[!,:ω] + 0.3*df[!,:k]
	# step 2
	nlobj = let
		# the let block here is for performance, see https://docs.julialang.org/en/v1/manual/performance-tips/#man-performance-captured
		f̂lag = panellag(f̂, df[!,id], df[!,t])
		klag = panellag(Matrix(df[!,fixedinputs]), df[!,id], df[!,t])
		inc = vec(all(.!ismissing.(f̂lag) .& .!ismissing.(klag) .& .!ismissing.(f̂), dims=2))
		println(length(inc))
		f̂lag = nomissing(f̂lag[inc]) # ensures type of elements is not a Union{Missing, T}, but just T
		klag = nomissing(klag[inc,:])
		f̂i = nomissing(f̂[inc])
		y=df[inc,output]
		k=Matrix(df[inc,fixedinputs])
		l=Matrix(df[inc,flexibleinputs])
		function nlobj(βk; degree=step2degree)
			ω̂lag = reshape(f̂lag .- klag*βk, length(y), 1)
			ω̂ = reshape(f̂i - k*βk, length(y),1)
			ỹ = reshape(y - l*βl - k*βk, length(y), 1)
			g = Dialysis.polyreg(ω̂lag, ω̂lag, ỹ, degree=degree)
			ξplusϵ = ỹ - g
			return(sum(ξplusϵ.^2)/length(ξplusϵ))
		end
	end
	β0 = fill((1-sum(βl)/(length(fixedinputs)+2)),
		length(fixedinputs))
	step2 = Optim.optimize(nlobj, β0, Optim.LBFGS(), autodiff=:forward)
	return(step1, step2, nlobj, eyex[:,1] - eyex[:,2:end]*βl)
end

end

# ╔═╡ 2ae11fa8-5797-11eb-0232-f3e685dfc51c
step1, step2, obj, f̂ = Estimators.olleypakes(df, :y, [:l], [:k], [:invest, :k], :id, :t, step1degree=4, step2degree=2)

# ╔═╡ e798a9fe-5a67-11eb-39ee-57c252ec2400
scatter(df[!,:ω], f̂ - df[!,:k]*p.βk, legend=:none, xlab="ω", ylab="ω̂")

# ╔═╡ 4765b3e4-582a-11eb-30f2-2b94983e172a
p

# ╔═╡ f15e4ee8-5757-11eb-1485-25a2bad8ac3d
βl, βk = step1.coef[2], step2.minimizer[1]

# ╔═╡ 466ca460-5826-11eb-3894-8f3cfce8f4c8
plot(β->obj([β]), 0, 1, legend=:none, xlabel="βₖ", ylabel="MSE")

# ╔═╡ 447914e0-5838-11eb-15f2-935c6721315f
md"""

!!! question
    What combinations of data generating processes and estimators produce good
    estimates? Explain why.
"""

# ╔═╡ Cell order:
# ╟─fa80b794-569d-11eb-2f16-0dad9fa54602
# ╠═dc247c6f-52fb-47ef-bd54-2c5a20cfe791
# ╠═8a4ade3b-eb29-4762-b32e-34adb96c9773
# ╠═16bfc249-ff90-45ea-a707-e96d59b30d84
# ╠═3ad05053-18cd-4e0a-92b3-0ced08d9a1f0
# ╠═d315f0e2-5754-11eb-2240-dd868564adfb
# ╠═2a662708-d5f0-41be-bc2e-56e00e2c975d
# ╠═1641e5b8-2a1f-4a37-85f1-84c4bab78932
# ╟─7d3e92a0-569d-11eb-3d2b-bdbfb6b4b364
# ╟─68f6e53e-56d4-11eb-3575-05ee3be38745
# ╟─4be9eee0-56dd-11eb-3458-e3ad0d5c1a78
# ╠═41769c24-567b-11eb-2f2d-9b51f92e7947
# ╠═55bc0887-d01d-4174-bb77-45e18989bffc
# ╠═90a815d7-f0fd-455c-a0b0-42eee3d91f6a
# ╠═8d1cfa5c-56a7-11eb-339e-3d733e6041ce
# ╠═a16f916a-5687-11eb-0219-7b16fa743194
# ╠═0437683c-5837-11eb-1f1c-33755302a653
# ╟─b5eeabfe-56dd-11eb-0292-43f1e2166311
# ╠═da42b1c0-56b4-11eb-2a47-531922e9e6d4
# ╠═d7d9043e-56b4-11eb-05af-1f838362a37d
# ╟─2d7f4818-5837-11eb-3679-dfaad86cd792
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
# ╟─21c66388-56e0-11eb-2e3c-618a1d2c0362
# ╠═6a2d413c-574c-11eb-39b5-cfca30ea3acc
# ╠═2ae11fa8-5797-11eb-0232-f3e685dfc51c
# ╠═e798a9fe-5a67-11eb-39ee-57c252ec2400
# ╠═4765b3e4-582a-11eb-30f2-2b94983e172a
# ╠═f15e4ee8-5757-11eb-1485-25a2bad8ac3d
# ╠═466ca460-5826-11eb-3894-8f3cfce8f4c8
# ╟─447914e0-5838-11eb-15f2-935c6721315f
