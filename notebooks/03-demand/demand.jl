### A Pluto.jl notebook ###
# v0.12.20

using Markdown
using InteractiveUtils

# ╔═╡ 9a669df8-6179-11eb-2a04-b9229c951c5a
using Revise, CodeTracking, BLPDemand, PlutoUI

# ╔═╡ 0af12182-6743-11eb-3c59-a3cbba8671e5
using Statistics, DataFrames

# ╔═╡ 038e7ae0-6746-11eb-21e0-3d708015b0ea
using Plots

# ╔═╡ 32dfd312-6178-11eb-2e72-f5f09e4cc4b6
md"""
# Demand Estimation
Paul Schrimpf

[UBC ECON567](https://faculty.arts.ubc.ca/pschrimpf/565/565.html)

[![Creative Commons License](https://i.creativecommons.org/l/by-sa/4.0/88x31.png)](http://creativecommons.org/licenses/by-sa/4.0/)
[Creative Commons Attribution-ShareAlike 4.0 International License](http://creativecommons.org/licenses/by-sa/4.0/)
"""

# ╔═╡ 8da42168-6178-11eb-0e5d-cfdfd8393b3c
md"""
This notebook will demonstrate some demand estimation methods. It focuses on the random coefficients logit model of demand for differentiated products of Berry, Levinsohn, & Pakes (1995).
"""

# ╔═╡ c21e0bc0-6178-11eb-1dad-cf2f0766862e
md"""
## Simulation

### Preferences

There are $J$ products with $K$ characteristics. Consumer $i$'s utility from product $j$ in market $t$ is:

```math
u_{ijt} = x_{jt}' \beta + x_{jt}' \Sigma \nu_{it} + \xi_{jt} + \epsilon_{ijt}
```

where $x_{jt}$ is a vector product characteristics, $\nu_{it}$ is a vector of unobserved idiosyncratic tastes for characteristics, $\xi_{jt}$ is an unobserved product demand shock, and $\epsilon_{ijt}$ is an independent Type-I extreme value shock.

### Market Shares

The market share of product $j$ in market $t$ is given by an integral over $\nu$ of logit probabilities:
```math
\sigma_j(x_t, \xi_t) = \int \frac{e^{x_{jt}' \beta + x_{jt}' \Sigma \nu + \xi_{jt}}}
{1 + \sum_{k=1}^J e^{x_{kt}' \beta + x_{kt}' \Sigma \nu_{it} + \xi_{kt}} } dF_\nu(\nu)
```
"""

# ╔═╡ 4b7ae8e6-6180-11eb-0e60-1969cb0229a7
md"""

### Profit Maximization

We will assume that firm's choose price, which will be the first element of $x$, ($x_{jt}[1]$), while other product characteristics are fixed.
Firms know the characteristics of all products and the whole vector $\xi_t$ when choosing $x_{jt}[1]$. Firms' profit maximization problem is:
```math
\max_{p} p M_t \sigma_j(x_t(p), \xi_t) - C_j(M_t \sigma_j(x_t(p),\xi_t))
```
where $x_t(p)$ denotes $x_t$ with $x_{jt}[1]$ set to $p$ and all other elements held fixed, $C_j$ is the cost function of the firm, and $M_t$ is the market size.

The firm's first order condition is
```math
\sigma_j(x_t(p),\xi_t) + p \frac{\partial \sigma}{\partial p}(x_t(p),\xi_t) =
C_j'(M_t \sigma_j(x_t(p), \xi_t)) \frac{\partial \sigma}{\partial p}(x_t(p),\xi_t)
```
Note that rearranging this equation gives the familiar price equals marginal cost plus markup formula:
```math
p_j - C_j' = -\frac{\sigma_j}{\partial \sigma_j/\partial p_j}
```
This is a mark**up** since $\partial \sigma_j/\partial p_j$ is generally negative.

### Marginal Costs

We will assume that marginal costs are log linear:
```math
C_j'(M s) = \exp( w_{jt}'\gamma + \omega_{jt})
```
where $w_{jt}$ are observed firm or product characteristics and $\omega_{jt}$ is an unobserved cost shock.


### Equilibrium

In equilibrium, each firm's price much satisfy the above first order condition given the other $J-1$ prices. Thus, equilibrium prices satisfy as system of $J$ nonlinear equations.

"""

# ╔═╡ 861ec9c8-6184-11eb-0d49-9b9e8573404c
md"""
### Simulating Data

We will use the [BLPDemand.jl](https://github.com/UBCECON567/BLPDemand.jl) package for computing and simulating the equilibrium. See the [docs](https://ubcecon567.github.io/BLPDemand.jl/dev/) for that package for more information.

In the simulation, exogenous product characteristics and cost shifters are $U(0,1)$. $\xi$ and $\omega$ are normally distributed.

To ensure that finite equilibrium prices exist, it is important that all consumers dislike higher prices, i.e. $\beta[1] + \nu[i, 1]*\sigma[1]<0$ for all $i$. Otherwise, the firm could charge an infinite price and just sell to the price loving consumers. Therefore, the first component of $\nu$ (the random coefficient on price) is $-U(0,1)$, and the other components of $\nu$ are normally distributed.
"""

# ╔═╡ 106eab00-61a8-11eb-0396-3973ff51f91e
md"""
To see the source for functions in `BLPDemand.jl` or any other package, you can use the `CodeTracking` package.
"""

# ╔═╡ 90dedeec-6a81-11eb-38e0-d12ed4abb84e


# ╔═╡ 65bff59c-61a6-11eb-1dfc-ff3419eb6a0b
md"""
The hardest part of simulating the model is solving for equilibrium prices. This is done by `eqprices` function, which uses the approach of [Morrow and Skerlos (2011)](https://doi.org/10.1287/opre.1100.0894).
"""

# ╔═╡ 7595f49a-674d-11eb-2b91-eba8579b0a96
md"""
We use the following parameters for simulating the data.
"""

# ╔═╡ 80dcc2d8-61a7-11eb-2c77-efa24935d94f
begin
	J = 5 # number of products
	T = 100 # number of markets
	β = [-0.1, # price coefficients
		ones(2)...] # other product coefficients
	σ = [0.5, ones(length(β)-1)...] # Σ = Diagonal(σ)
	γ = ones(1) # marginal cost coefficients
	S = 10 # number of Monte-Carlo draws per market and product to integrate out ν
	sξ = 0.2 # standard deviation of demand shocks
	sω = 0.2 # standard deviation of cost shocks
end;

# ╔═╡ 4b319506-617b-11eb-3de0-e1f2ba8d083f
let
	# get the function definition
	str = @code_string simulateBLP(J, T, β, σ, γ, S)

	# for nicer display in Pluto
	str = "```julia\n"*str*"\n```\n"
	Markdown.parse(str)
end

# ╔═╡ 8b218732-61a8-11eb-1304-95b305205fc3
let
	# we need to provide arguments to eqprices, so that @code_string knows the types of the arguments, so it can look up the desired method, but the contents of the arguments don't matter; just their types
	str = @code_string eqprices(ones(2), β, σ, ones(2), ones(2,2), ones(2,2))
	str = "```julia\n"*str*"\n```\n"
	Markdown.parse(str)
end

# ╔═╡ 97391104-670c-11eb-1bdd-0577f09c3a47
sim = simulateBLP(J, T, β, σ, γ, S, varξ=sξ, varω=sω);

# ╔═╡ 01554514-6a67-11eb-2f58-395e4703d401
begin
	using Optim, LineSearches
	PlutoUI.with_terminal() do
		nfxp = estimateBLP(sim.dat, method=:NFXP, verbose=true,
			  			   optimizer = LBFGS(linesearch=LineSearches.HagerZhang()))
	end
end

# ╔═╡ 06c04b06-6741-11eb-1fb3-effcbc26fbbe
typeof(sim)

# ╔═╡ 437777fe-674d-11eb-29a7-4d0f816adef4
md"""
We can see that `simulateBLP` returns a named tuple. The first element of the tuple is an array of `MarketData` which represents data that we would observe. The next two are the unobserved demand and cost shocks.

`MarketData` is a `struct` defined in `BLPDemand.jl`.

Let's look at some summary statistics and figures describing the simulated data.
"""

# ╔═╡ 762289f6-6742-11eb-09cb-19e7a296fe0c
begin
	function describe(d::Array{MarketData})
		T = length(d)
		funcs = [minimum, maximum, mean, std]
		cnames = ["min","max","mean","std dev"]
		rnames = ["Number of Products", "Shares", "Prices",
				"Other Characteristics",
				"Cost Shifters",
				"Demand Instruments",
				"Supply Instruments"]
		trans = [x->length(x.s),
				x->x.s,
				x->x.x[1,:],
				x->x.x[2:end,:],
				x->x.w,
				x->x.zd,
				x->x.zs]
		df = DataFrame([[f(vcat(t.(d)...)) for t in trans] for f in funcs])
		rename!(df, cnames)
		insertcols!(df, 1,Symbol(" ") => rnames)
		df
	end
	describe(sim.dat)
end

# ╔═╡ 5b80e12a-6748-11eb-0a2c-3d272d21c92a
let
	Plots.gr(fmt="png")
	j = 1
	k =1
	fig=scatter(vcat( (x->x.x[1,j]).(sim.dat)...),
		vcat( (x->x.s[k]).(sim.dat)...),
		 ylabel="Share of Product $k", xaxis=:level)
	plot!(fig, xlabel="Price of Product $j")
	fig
end

# ╔═╡ f4091954-6745-11eb-1acf-f9878b37d78c
let
	j = 1
	k =1
	scatter(vcat( (x->x[j]).(sim.ξ)...),
		vcat( (x->x.s[k]).(sim.dat)...),
		xlabel="ξ[$j]",
		ylabel="Share of Product $k")
end

# ╔═╡ e4b9cc8a-6751-11eb-3e0b-01ea772ac1b3
md"""
## Estimation

It is conventional to group together all product and market specific terms entering consumers' utility as
```math
\delta_{jt} = x_{jt}' \beta + \xi_{jt}
```
Let $\delta_t$ denote the vector $J$ values of $\delta_{jt}$ in market $t$.

Define the share as a function of $\delta_t$ as
```math
\sigma_{jt}(\delta_t;\theta) =  \int \frac{e^{\delta_jt + x_{jt}' \Sigma \nu}}
{1 + \sum_{k=1}^J e^{\delta_{kt} + x_{kt}' \Sigma \nu_{it} }}  dF_\nu(\nu)
```
where $\theta$ denotes all the parameters to be estimated.

Let $\sigma_t(\delta_t;\theta)$ denote the vector of $J$ shares.

Define the inverse of this function as $\Delta_t(s_t;\theta) = d$ where $d$ satisfies $\sigma_t(d;\theta) = s_t$.

The demand-side moment condition for this model can then be written as
```math
E[(\underbrace{\Delta_{jt}(s_t;\theta) - x_{jt}\beta}_{\xi_jt}) z^D_{jt} ] = 0
```

Using the log-linear marginal specification, we can rearrange the first order condition above to isolate the cost shock:
```math
\omega_{jt} = \log\left(p_{jt} + \frac{\sigma_{jt}(\Delta_t(s_t;\theta);\theta)}{
\frac{\partial \sigma_{jt}}{\partial p_j}(\Delta_t(s_t;\theta);\theta)} \right) - w_{jt}' \gamma
```
and the supply side moments can be written as:
```math
E\left[\underbrace{\left(\log\left(p_{jt} + \frac{\sigma_{jt}(\Delta_t(s_t;\theta);\theta)}{
\frac{\partial \sigma_{jt}}{\partial p_j}(\Delta_t(s_t;\theta);\theta)} \right) - w_{jt}' \gamma\right)}_{\omega_{jt}} z_{jt}^S\right] = 0
```
To more succintly denote the moments, define:
```math
g_{jt}(\delta, \theta) = \begin{pmatrix} (\delta_{jt} - x_{jt} \beta)z_{jt}^D
\\
\left( \log\left(p_{jt} + \frac{\sigma_{jt}(\delta_t;\theta)}{
\frac{\partial \sigma_{jt}}{\partial p_j}(\delta_t;\theta)} \right) - w_{jt}' \gamma\right) z_{jt}^S
\end{pmatrix}
```

Then the combined moments conditions can be written as
```math
E[g_{jt}(\Delta(s_t; \theta),\theta) ]  = 0
```

To esitmate $\theta$ we minimize a quadratic form in the empirical moments:
```math
\hat{\theta} \in \mathrm{arg}\min_\theta \left(\frac{1}{JT} \sum_{j,t} g_{jt}(\Delta(s_t;\theta),\theta) \right)' W \left(\frac{1}{JT} \sum_{j,t} g_{jt}(\Delta(s_t;\theta),\theta) \right)
```

"""


# ╔═╡ b45bd3c4-6a6e-11eb-0004-3f0ebdec62e0


# ╔═╡ 690b6616-6a62-11eb-2dbf-c9699fc9785f
md"""
## Nested Fixed Point

The original BLP paper estimated $\theta$ using a "nested fixed point" (NFXP) approach. The function $\Delta(s; \theta)$ has no closed form expression.  $\Delta(s;\theta)$ is the solution to $\sigma(\Delta(s;\theta);\theta) = s$. This equation can only be solved numerically with some sort of iterative algorithm. BLP rearrange the equation to write $\delta$ as fixed point the point of a contraction mapping, $\Delta(s;\theta) = F(\Delta(s;\theta),\theta,s)$ and then compute $\Delta$ iteratively:

- Choose $d_0$ arbitrarily
- Define $d_i = F(d_{i+1},\theta,s)$ for $i = 1, 2, ...$
- Continue until $\Vert d_i - d_{i-1} \Vert$ is small enough
- Use the final $d_i$ as $\Delta(s,\theta)$

This simple repeated application of a contraction mapping is robust, but not neccessarily the most efficient way to compute $\Delta$. Other methods can compute $\Delta$ more quickly, but can be less stable. All methods involve some of iteration until convergence. The function `delta` in `BLPDemand.jl` computes $\Delta$.

The "nested" part of nested fixed point refers to the fact that minimizing the GMM objective function also involves some sort of [iterative minimization algorithm](https://schrimpf.github.io/AnimatedOptimization.jl/optimization/#optimization-algorithms).  Each iteration of the minimization algorithm, we redo all the iterations needed to compute $\Delta(s,\theta)$.

"""

# ╔═╡ 597b178e-674a-11eb-3a2e-e9a363c00a85
md"""
!!! info "Automatic Differentiation"
    We need to compute deriviatives of many of the functions above. $\frac{\partial \sigma}{\partial p}$ is part of the moment conditions. Minimization algorithms can be much more efficient if we provide the gradient and/or hessian of the objective function. Finally, the asymptotic variance of $\theta$ involves the Jacobian of the moment conditions (i.e. $D_\theta g_{jt} (\Delta(s_t;\theta),\theta) $). While we could calculate these derivatives by hand and then write code to compute them, there is a less laborious and error-prone way. Automatic differentiation refers to taking code that computes some function, $f(x)$ and algorithmically applying the chain rule to compute $\frac{df}{dx}$. One of Julia's nicest features is how well it supports automatic differentiation. The two leading Julia packages for automatic differentiation are [ForwardDiff.jl](https://github.com/JuliaDiff/ForwardDiff.jl) and [Zygote.jl](https://github.com/FluxML/Zygote.jl). ForwardDiff.jl is usually a good choice for function from $R^n \to R^k$ with $n$ not too much larger than $k$. Zygote.jl is somewhat more restrictive in what code it is compatible with, but is more efficient if $n$ is much much larger than $k$. You can read more in the [Quantecon notes on automatic differentiation](https://julia.quantecon.org/more_julia/optimization_solver_packages.html).
"""

# ╔═╡ fd33a886-6a6b-11eb-3e5f-655d4a95cfbf
md"""

## MPEC

MPEC is an alternative approach to computing estimates promoted by Judd and Su and applied to this model by Dube, Fox, and Su (2012). MPEC stands for "Mathematical Programming with Equilibrium Constraints." It is based on the observation that the estimation problem can be expressed as a constrained minimization problem

```math
(\hat{\theta},\hat{\delta}) = \mathrm{arg}\min_{\theta, \delta}
 \left(\frac{1}{JT} \sum_{j,t} g_{jt}(\delta_t,\theta) \right)' W \left(\frac{1}{JT} \sum_{j,t} g_{jt}(\delta_t,\theta) \right) \text{ s.t. } s_t = \sigma_t(\delta_t;\theta) \;\forall t
```

Note that this problem no longer requires compute $\Delta(s,\theta)$, so there is no more nesting of iterative computations. However, there are some tradeoffs. We now have a constrained minimization problem, which are slightly more complicated to solve than unconstrained problems. More importantly, the minimization problem is now over a much higher dimensional space ($\theta$ and $\delta$ instead of just $\theta$). Nonetheless, the MPEC formulation of the problem can sometimes lead to more efficient and/or more robust computation. The relative performance of MPEC and NFXP is very dependent on the implementation details (the minization algorithms, the way derivatives are calculated, and whether any sparsity patterns are taken advantage of).

"""

# ╔═╡ 48df1e86-6a6f-11eb-1fa4-018317fc7664
PlutoUI.with_terminal() do
	mpec = estimateBLP(sim.dat, method=:MPEC, verbose=true)
end

# ╔═╡ a578173c-6a6b-11eb-0171-43b29df3bbbb


# ╔═╡ Cell order:
# ╟─32dfd312-6178-11eb-2e72-f5f09e4cc4b6
# ╟─8da42168-6178-11eb-0e5d-cfdfd8393b3c
# ╟─c21e0bc0-6178-11eb-1dad-cf2f0766862e
# ╟─4b7ae8e6-6180-11eb-0e60-1969cb0229a7
# ╟─861ec9c8-6184-11eb-0d49-9b9e8573404c
# ╟─106eab00-61a8-11eb-0396-3973ff51f91e
# ╠═9a669df8-6179-11eb-2a04-b9229c951c5a
# ╠═90dedeec-6a81-11eb-38e0-d12ed4abb84e
# ╠═4b319506-617b-11eb-3de0-e1f2ba8d083f
# ╟─65bff59c-61a6-11eb-1dfc-ff3419eb6a0b
# ╟─8b218732-61a8-11eb-1304-95b305205fc3
# ╟─7595f49a-674d-11eb-2b91-eba8579b0a96
# ╠═80dcc2d8-61a7-11eb-2c77-efa24935d94f
# ╠═97391104-670c-11eb-1bdd-0577f09c3a47
# ╠═06c04b06-6741-11eb-1fb3-effcbc26fbbe
# ╟─437777fe-674d-11eb-29a7-4d0f816adef4
# ╠═0af12182-6743-11eb-3c59-a3cbba8671e5
# ╠═762289f6-6742-11eb-09cb-19e7a296fe0c
# ╠═038e7ae0-6746-11eb-21e0-3d708015b0ea
# ╠═5b80e12a-6748-11eb-0a2c-3d272d21c92a
# ╠═f4091954-6745-11eb-1acf-f9878b37d78c
# ╟─e4b9cc8a-6751-11eb-3e0b-01ea772ac1b3
# ╟─b45bd3c4-6a6e-11eb-0004-3f0ebdec62e0
# ╟─690b6616-6a62-11eb-2dbf-c9699fc9785f
# ╟─597b178e-674a-11eb-3a2e-e9a363c00a85
# ╠═01554514-6a67-11eb-2f58-395e4703d401
# ╟─fd33a886-6a6b-11eb-3e5f-655d4a95cfbf
# ╠═48df1e86-6a6f-11eb-1fa4-018317fc7664
# ╟─a578173c-6a6b-11eb-0171-43b29df3bbbb
