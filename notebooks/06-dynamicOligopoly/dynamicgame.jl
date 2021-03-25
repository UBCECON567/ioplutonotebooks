### A Pluto.jl notebook ###
# v0.12.21

using Markdown
using InteractiveUtils

# ╔═╡ b7f912b2-8b48-11eb-080e-c3edaf9b55c4
using PlutoUI, Plots, Statistics, StatsPlots, Random, PrettyTables, DataFrames, Printf

# ╔═╡ ed50e4f4-8a84-11eb-2e5b-075cfdea1280
module DG

using NLsolve, LinearAlgebra, Distributions

export DynamicGame, equilibrium, Λ, vᵖ

struct DynamicGame{I<:Integer, U<:Function, R<:Real, E<:Function}
	"N is the number of players"
	N::I
	
	"""
	u(i, a, x) is the flow payoff function, where 
	- i is a player index
	- a is a vector of actions of length N
	- x is a state
	"""
	u::U
	
	"Discount factor"
	β::R
	
	"""
	A function returning a vector with Ex(a,x)[x̃] = P(x̃|a,x)
	"""
	Ex::E
	
	"""
	Set of actions, must be 1:A or similar.
	"""
	actions 
	
	"""
	Set of states, must be 1:S or similar.
	"""
	states
end

function pmax(v::AbstractVector)
	m = maximum(v)
	p = exp.(v .- m)
	p ./= sum(p)
	return(p)
end

function emax(v::AbstractVector)
	m = maximum(v)
	return(Base.MathConstants.γ + m + log(sum(exp.(v .- m))))
end

"""
   vᵖ(g::DynamicGame, P)

Computes choice specific value functions given strategies P.

P should be an array with P[i,a,x] = P(aᵢ=a| x)
"""
function vᵖ(g::DynamicGame, P)
	v = similar(P)
	a_all = Vector{eltype(g.actions)}(undef, g.N)
	E = zeros(eltype(P), length(g.actions), length(g.states), 
		length(g.actions), length(g.states))
	y = zeros(eltype(v), length(g.actions), length(g.states))	
	for i in 1:g.N
		y .= zero(eltype(y)) 
		E .= zero(eltype(E))
		for a0 in g.actions
			a_all[i] = a0		
			for x in g.states
				for ami in Iterators.product(ntuple(i->g.actions, g.N-1)...)
					Pmi = one(eltype(P))
					for j in 1:g.N
						if j ≠ i
							k = j < i ? j : (j-1)
							Pmi *= P[j, ami[k], x]
							a_all[j] = ami[k]
						end
					end
					y[a0,x] += Pmi*g.u(i,a_all,x)
					for an in g.actions 
						E[a0,x,an,:] .+= Pmi*g.Ex(a_all,x).*P[i,an,:]
						y[a0,x] +=  Pmi*g.β * (g.Ex(a_all,x).*P[i,an,:])'*
							(-log.(P[i,an,:]) .+ Base.MathConstants.γ)
					end
				end # ami
			end # x
		end # an
		SA = length(g.states)*length(g.actions)
		v[i, :, : ] .= reshape( (I - g.β*reshape(E, SA, SA)) \ reshape(y, SA), 
							   length(g.actions), length(g.states)) 
	end # i
	return(v)
end

"""
	V̄(g::DynamicGame, vᵖ)

Returns the value function for game `g` with choice specific value functions vᵖ.
"""
function V̄(g::DynamicGame, vᵖ)
	return([emax(vᵖ[i,:,x]) for i in 1:g.N, x in g.states])
end

"""
    Λ(g::DynamicGame, vᵖ)

Computes best response choice probabilities given choice specific value function.
"""
function Λ(g::DynamicGame, vᵖ)
	p = similar(vᵖ)
    for (i,x) in Iterators.product(1:g.N, g.states)
		p[i,:,x] .= pmax(vᵖ[i,:,x])
	end
	return(p)
end


"""
    equilibrium(g::DynamicGame)

Compute equilibriumn choice probabilites of game `g`.

Returns a tuple `(out, P)` where `out` is the return value of `nlsolve`, and the choice probabilities are `P`.
"""
function equilibrium(g::DynamicGame)
	p = zeros(g.N,length(g.actions), length(g.states))
	#p = rand(size(p)...)
	#p .= 1/length(g.actions)	
	p[:,1,:] .= 0.1
	p[:,2,:] .= 0.9
	function probs(z)
		p = similar(z, size(z,1), size(z,2) + 1, size(z,3))
		ez = exp.(z)
		for i in 1:size(p,1)
			for x in 1:size(p,3)
				se = sum(ez[i,:,x])
				p[i,2:end,x] .= ez[i,:,x]./(1 + se)
				p[i,1,x] = 1/(1 + se)
			end
		end
		return(p)
	end		
	z = log.(p[:,2:end,:]) 
	for c in 1:size(z,2)
		z[:,c,:] .-= log.(p[:,1,:])
	end
	function eq!(e,z)
		p = probs(z)		
		e .= (p - Λ(g, vᵖ(g, p)))[:,2:end,:]
		return(e)
	end
	out = nlsolve(eq!, z, autodiff=:forward, method=:trust_region)
	return(out, probs(out.zero))
end

"""
    simulate(g::DynamicGame, T, P; burnin=T, x0=rand(g.states))

Simulates game `g` for `T` periods with strategies `P`. Begins from state `x0` and discards the first `burnin + 1` periods.
"""
function simulate(g::DynamicGame, T, P; burnin=T, x0=rand(g.states))
	A = similar(g.actions,g.N,T)
	U = zeros(g.N,T)
	EV = copy(U)
	V = copy(U)
	X = similar(g.states,T)
	x = copy(x0)
	a = similar(g.actions, g.N)
	v = vᵖ(g, P)
	for t=-burnin:T
		ϵ = rand(Gumbel(0,1),g.N,length(g.actions))
		for i in 1:g.N
			#(_, a[i]) = findmax(v[i,:,x] + ϵ[i,:])			
			(_, a[i]) = findmax(log.(P[i,:,x]).-log(P[i,1,x]) + ϵ[i,:])
		end		
		if (t>0)
			A[:,t] .= a
			X[t] = x			
			for i in 1:g.N
				u = g.u(i,a,x)
				U[i,t] = u + ϵ[i,a[i]]
				V[i,t] = v[i,a[i],x] + ϵ[i,a[i]]
				EV[i,t] =v[i,a[i],x] - log.(P[i,a[i],x]) + Base.MathConstants.γ
			end
		end
		x = rand(DiscreteNonParametric(g.states, g.Ex(a,x)))
	end
	return(a=A, x=X, u=U, v=V, ev=EV)
end

end

# ╔═╡ 93acf058-8c25-11eb-1cb5-d391f007119c
"Dynamic game estimation."
module DGE

using LinearAlgebra,  Statistics, Distributions

function choiceprob(data)
	states = sort(unique(data.x))
	actions = sort(unique(data.a))
	N = size(data.a,1)	
	return([sum( (data.a[i,:].==a) .& (data.x.==x)) /
			sum(data.x.==x) for i ∈ 1:N, a ∈ actions, x ∈ states])
end

function transitioni(data)
	# P(x'|a_i,x,i)	
	states = sort(unique(data.x))
	actions = sort(unique(data.a))
	N  = size(data.a,1)
	Pxi = [sum((data.x[2:end].==x̃) .& (data.x[1:(end-1)].==x) 
			   .& (data.a[i,1:(end-1)].==a)) /
		   sum((data.x[1:(end-1)].==x) .& (data.a[i,1:(end-1)].==a) )
		   for i ∈ 1:N, x̃ ∈ states, a ∈ actions, x ∈ states]
	return(Pxi)
end

function constructu(data, β; P=choiceprob(data), Pxi=transitioni(data), a0=1)
	states = sort(unique(data.x))
	actions = sort(unique(data.a))
	N  = size(data.a,1)
	Eu = similar(P)
	Eu[:,a0,:] .= 0
	v = similar(P)
		
	# recover v[:,a0,:]
	for i ∈ 1:N 
		q = [Base.MathConstants.γ - log(P[i,a0,x]) for x ∈ states]
		E = Pxi[i,:,a0,:]'
		y = Eu[i,a0,:] + β*E*q
		v[i,a0,:] .= (I - β*E) \ y
	
		for a ∈ actions
			if a ≠ a0
				v[i,a,:] .= log.(P[i,a,:]) .- log.(P[i,a0,:]) .+ v[i,a0,:]
			end			
		end
		# recover E[u(a[i],a[-i],x)|a[i],x]
		q = [Base.MathConstants.γ + log(sum(exp.(v[i,:,x]))) for x ∈ states]
		for a ∈ actions
			E = Pxi[i,:,a,:]'
			Eu[i,a,:] .= v[i,a,:] .- β*E*q
		end
	end
	return(Eu=Eu, v=v)	
end


function transition(data)
	states = sort(unique(data.x))
	actions = sort(unique(data.a))
	N  = size(data.a,1)	
	if N != 2
		error("transition assumes 2 players")
	end
	Px = [sum( (data.x[2:end] .== x̃) .& 
			(data.a[1,1:(end-1)].==a1) .& 
			(data.a[2,1:(end-1)].==a2) .& 
			(data.x[1:(end-1)].==x)) /
		 sum((data.a[1,1:(end-1)].==a1) .& 
			(data.a[2,1:(end-1)].==a2) .& 
			(data.x[1:(end-1)].==x)) for x̃ ∈ states, 
			a1 ∈ actions, a2 ∈ actions, x ∈ states ]
	Px[isnan.(Px)] .= 0 ./ length(states)
	return(Px)
end

		
function markovbootstrap(data, P = choiceprobs(data), Px=transition(data))
	states = sort(unique(data.x))
	actions = sort(unique(data.a))
	N  = size(data.a,1)	
	bd = deepcopy(data)
	T = length(data.x)
	for t ∈ 2:T
		bd.x[t] = rand(DiscreteNonParametric(states, Px[:,bd.a[:,t-1]...,bd.x[t-1]]))
		for i in 1:N
			bd.a[i,t] = rand(DiscreteNonParametric(actions,P[i,:,bd.x[t]]))
		end
	end
	return(bd)
end
			
end			
			

# ╔═╡ dd4c95aa-8a82-11eb-3be1-9309fb5be5af
md"""
# Dynamic Oligopoly

Paul Schrimpf

[UBC ECON567](https://faculty.arts.ubc.ca/pschrimpf/565/565.html)

[![Creative Commons License](https://i.creativecommons.org/l/by-sa/4.0/88x31.png)](http://creativecommons.org/licenses/by-sa/4.0/)
[Creative Commons Attribution-ShareAlike 4.0 International License](http://creativecommons.org/licenses/by-sa/4.0/)
"""

# ╔═╡ 596b18be-8a83-11eb-01fe-07833e8ac25e
md"""
This notebook will simulate and estimate some models of dynamic oligopoly.
"""

# ╔═╡ 6a6bbfc4-8a83-11eb-308b-0d1d918f30f1
md"""
# Model

The model and notation are the same as in [the course slides](https://faculty.arts.ubc.ca/pschrimpf/565/06-dynamicOligopoly.pdf). 

The state of the game is $x \in \texttt{states}$. 

Each period, players simultaneous choose $a \in \texttt{actions}$. The flow payoff for player $i$ is

```math
u(i, a, x) + \epsilon_i[a]
```

where $\epsilon_i[a]$ are i.i.d. Gumbel(0,1) distributed (as in a multinomial logit).


## Choice specific values

To compute the equilibrium and estimate the model, we will work choice specific value functions as functions of choice-probabilities. Given choice probabilities, $P[i,a,x] = P(aᵢ==a|x)$, define choice specific value functions as the solution to
```math
v^p[i,a_i,x] = \sum_{a_{-i}} P(a_{-i}|x) \left( u(i,a_i, a_{-i},x) + \beta \sum_{x'} P(x'|a_i, a_{-i}, x) \sum_{a'} P[i,a',x'] (v^p[i, a', x'] + E\left[\epsilon[a']\,\vert\,v^p[i, a', x'] + \epsilon[a'] \geq v^p[i, \tilde{a}, x'] + \epsilon[\tilde{a}]\right] \right)
```

Note that the last term in the above expression can pe written as a function of choice probabilities and the distribution of $\epsilon$. For the Gumbel distribution, it becomes ([see here for a derivation](https://stats.stackexchange.com/a/192495/1229)):
```math
E\left[\epsilon[a']\,\vert\,v^p[i, a', x'] + \epsilon[a'] \geq v^p[i, \tilde{a}, x'] + \epsilon[\tilde{a}]\right] = \gamma - \log(P[i,a',x'])
```
This leaves us with a linear system of equations to solve for $v^P$ given $P$. We do this in the `vᵖ` function below. 

Conversely, given choice specific values, we can compute implied choice probabilities as simply
```math
\begin{align*}
P[i,a,x] & = P\left(v^p[i, a, x] + \epsilon[a] \geq v^p[i, \tilde{a}, x] + \epsilon[\tilde{a}]\right) \\
& = \frac{e^{v^p[i, a, x]}}{\sum_{\tilde{a}} e^{v^p[i, \tilde{a}, x]}}
\end{align*}
```

## Equilibrium

Let $\Lambda(v^P)$ denote the mapping from choice specific values to choice probabilities. Then an equilibrium vector of choice probabilities statisfies
$P = \Lambda(v^P)$. We use the `NLsolve.jl` to numerically solve this system of equations.

"""




# ╔═╡ 8026965a-8b90-11eb-19aa-b17e1eff6034
md"""
!!! question
    Are modules a good idea inside pluto? I'm not sure. A downside is that it make editing require more scrolling up and down. An upside is that it will make it slightly easier to put this code into a package if I ever want to.
"""

# ╔═╡ 2451bac4-8b8d-11eb-0a68-53f3fd21ffad
md"""

## Payoffs and Transition Probabilities

The following code defines the payoff function and transition probabilities of the states. 

Each firm chooses action 1 or 2. The state consists of each firms' previous action, and `Nexternal` binary values that evolve exogenously. 

The payoff function could be thought of as some kind of entry/exit game. Action 1 is exit, 2 in entry (or continued operation). The payoff of exit is 0. The payoff of operating consists of revenues if you entered/continued operation last period, minus a cost of operation, minus an additional entry cost. The revenues of operating decline with the number of other firms operating, and increase with the external states.

"""

# ╔═╡ dfbf24ee-8a97-11eb-368f-555cd55c151e
(N, ns, u, Ex, statevec, stateind, states) = let
	N = 2
	Nexternal=1
	
	# There's often some tedious book keeping involved in going from an integer state index to a vector representation of a state 
	states = BitVector.(digits.(0:(2^(N+Nexternal)-1), base=2, pad=N+Nexternal))
	statevec(x::Integer)=states[x]
	stateind(s::AbstractVector)=findfirst([s==st for st in states])
	u(i, a, x::Integer) = u(i,a,statevec(x))
	function u(i, a, s::AbstractVector)
		return( (a[i]-1)*s[i]*(3-Nexternal/2 + sum(s[(N+1):end]) - sum(s[1:N]))
			- 0.5*(a[i]-1)*(1-s[i]) # entry cost
			- s[i]*(0.8 + 0.1*sum(s[1:N]))*(a[i]-1)) # fixed cost
	end

	Ex(a, x::Integer) = Ex(a, statevec(x))
	pstay = 0.7 # each binary external state stays the same with probability pstay
    function Ex(a, s::AbstractVector)
		E = zeros(length(states))
		sn = copy(s)
		sn[1:N] .= a.-1
		for j in 0:(2^Nexternal-1)
			sn[(N+1):end] .= digits(j, base=2, pad=Nexternal)
			i = stateind(sn)
			nsame = sum(sn[(N+1):end].==s[(N+1):end])
			E[i] = pstay^(nsame)*(1-pstay)^(Nexternal-nsame)
		end
		return(E)
	end	
	N, length(states), u, Ex, statevec, stateind, states
end



# ╔═╡ 7f7ecd96-8b93-11eb-0505-0925707e55c2
md"""
## Compute the Equilibrium

This code should work well with 2 players. I believe the payoffs are such that actions are strategic subsititutes, and I believe the equilibrium should be unique. 

I would expect there to be numeric difficulties in some cases with more players and/or a modified payoff function.


"""

# ╔═╡ 81dcea24-8b24-11eb-0c49-3bc494f41eb8
begin
	g = DG.DynamicGame(N, u, 0.9, Ex, 1:2, 1:ns)
	out = DG.equilibrium(g)
end

# ╔═╡ 3589d3b8-8b94-11eb-0351-9d70e67fc6ef
md"""
## Checking the Equilibrium

It is always a good idea to test your code. 
It's especially important here since the code above is fairly complicated. 

We first just print some the equilibrium choice probabilities and value functions.
"""

# ╔═╡ 4d99c88a-8b2f-11eb-12b0-47811a6bbed3
let
	p = copy(out[2])
	#p .= 1/2
	PlutoUI.with_terminal() do
		v = DG.vᵖ(g, p)
		for x in g.states
			@show x,statevec(x)
			for i in 1:g.N
				@show p[i,:,x]			
				@show v[i,:,x]
			end
			#@show p[2,:,x]
			#@show v[2,:,x]
		end
	end
end

# ╔═╡ 83ad2cc2-8b94-11eb-2ef6-a59ccdb501f3
md"""
Now we will simulate the model and verify that the model value functions match the average discounted sum of payoffs in the simulations.
"""

# ╔═╡ 4c7340f6-8b77-11eb-2f62-9f3f6422d95e
sim = DG.simulate(g, 1000, out[2], burnin=0, x0=1)

	

# ╔═╡ a441b890-8b94-11eb-1817-c995add52fa8
md"""
The output below shows the probability of each state in the simulated data, the value function of player 1 in that state, and the average discounted payoffs of player 1 in that state. These last two numbers in each row should be approximately equal. 
"""

# ╔═╡ 9a6e598c-8b7a-11eb-0b29-c57ecadec0aa
let
	v = DG.vᵖ(g, out[2])
	V̄ = DG.V̄(g, v)
	sumu = zeros(eltype(sim.u),g.N,length(g.states))
	T = length(sim.x)
	for x in unique(sim.x)
		t0 = findall(x.==sim.x)
		for i in 1:g.N
			sumu[i,x] = mean( [sum(sim.u[i,t:end].*(g.β.^(0:(T-t)))) for t in t0] )
		end
	end
	[(x,mean(sim.x.==x), V̄[1,x],sumu[1,x]) for x in g.states]
end
	

# ╔═╡ 1a253676-8beb-11eb-1d16-170c65346d9f
md"""
We can also check that deviating from the equilibrium choice probabilities lowers average payoffs.
"""

# ╔═╡ 348a21de-8beb-11eb-027f-5d519ea12572
let
	T = 4000
	seed = reinterpret(Int, time())
	Random.seed!(seed)
	P = copy(out[2])
	s0 = DG.simulate(g, T, out[2] , burnin=0, x0=1)
	Random.seed!(reinterpret(Int, time()))
	dp = min(0.5,minimum(P[1,:,:]))*(2*rand(size(P,3)).-1)
	P[1,1,:] .-= dp
	P[1,2,:] .+= dp
    Random.seed!(seed)
	s1 = DG.simulate(g, T, P, burnin=0, x0=1)
	
	function EV(sim)
		ev = zeros(g.N,length(g.states))
		ev2 = zeros(g.N,length(g.states))
		for x in unique(sim.x) #, a in g.actions
			t0 = findall((x.==sim.x))# .& (a.==sim.a[1,:]))
			for i in 1:g.N
				ev[i,x] = mean( [sum(sim.u[i,t:end].*(g.β.^(0:(T-t)))) for t in t0] )
			end
		end
		return(ev)
	end
	(mean(s0.a .!= s1.a), EV(s0)[1,:] - EV(s1)[1,:])
end

# ╔═╡ 27ab599c-8c06-11eb-36c3-090611cb8458


# ╔═╡ 94110c2a-8c03-11eb-3d00-ebab8f1d3811
[t:(t+9) for t in 1:10:100]

# ╔═╡ 0518cb98-8bec-11eb-1c55-e19103432968
md"""
It would be a good idea to automate these checks as [unit tests](https://docs.julialang.org/en/v1/stdlib/Test/).
"""

# ╔═╡ 8dfbd0c6-8b75-11eb-3643-05334135f3ce
md"""
## Plots

We now create some plots of the simulated data.

First, a time series plot of actions and the state.
"""

# ╔═╡ 89d97f3a-8be2-11eb-3faf-ad0a6dc22571
let 
	t = 1:100
	f1=plot(t,[sum(statevec(x)[3:end]) for x in sim.x[t]], ylabel="External state", legend=:none)
	f2=plot(t,sim.a[1,t], ylabel="Player 1 Action", legend=:none)
	f3=plot(t,sim.a[2,t], ylabel="Player 2 Action", legend=:none)
	plot(f1,f2,f3,layout=(3,1))
end
		

# ╔═╡ 5351f0d0-8be4-11eb-05da-75dfb199544f
let
	S = length(g.states)
	Px = ones(S,S)
	for x in unique(sim.x), x̃ in unique(sim.x)
		Px[x̃,x] = sum((sim.x[1:(end-1)].==x) .& (sim.x[2:end].==x̃)) / 
		           sum(sim.x[1:(end-1)].==x) 
	end
	heatmap(1:S,1:S,Px, xlab="Initial State", ylab="Next State", 
		title="P(Next State | Initial State)",
	 	c=cgrad([:white,:red]))
	#plot(1:100, sim.x[1:100])
end

# ╔═╡ 0af9d260-8be6-11eb-2d51-5f803d5456bf
let
	S = length(g.states)
	A = length(g.actions)
	Pa = zeros(g.N,A,S)
	for x in g.states, a in g.actions, i in 1:g.N
		Pa[i,a,x] = sum((sim.x.==x) .& (sim.a[i,:].==a)) / 
		           sum(sim.x.==x) 
	end
	f1 = heatmap(g.states,g.actions,Pa[1,:,:], xlab="State", ylab="Action", 
		title="P(Action|State, i=1)",
	 	c=cgrad([:white,:orange]),
		yticks=(1:2, ["out","in"]),
		xticks=(1:S, (x->"$x").(statevec.(1:S))),
		xrotation=45,
		tickfontsize=8)
	if (g.N > 1)
	f2 = heatmap(1:S,1:A,Pa[2,:,:], xlab="State", ylab="Action", 
		title="P(Action|State, i=2)",
	 	c=cgrad([:white,:orange]),
		yticks=(1:2, ["out","in"]),
		xticks=(1:S, (x->"$x").(statevec.(1:S))),
		xrotation=45,
		tickfontsize=8)
	plot(f1,f2, layout=(2,1))	
	else
		f1
	end
end

# ╔═╡ 2732717a-8c07-11eb-3a00-d19cfae02477
let 
	S = length(g.states)
	histogram(sim.x, legend=:none, xlab="state",
		xticks=(1:S, (x->"$x").(statevec.(1:S))),
		xrotation=45,
		tickfontsize=8)
end

# ╔═╡ 5b3a2268-8c07-11eb-31bb-a320fd9e7bb0
let 
	histogram(vec(sim.a), legend=:none, xlab="action", xticks=1:2)
end

# ╔═╡ 647103a0-8b95-11eb-29bc-6d26c83d4a13
md"""
# Estimation

We will estimate the model by following the steps of the identification proof in the slides. That identification argument is constructive --- it explicitly states how to compute the payoff function given choice and transition probabilities. This estimation approach has the advantage of being numerically stable. However, it might not be as statistically efficient as a maximum likelihood or method of moments estimator.


"""

# ╔═╡ 1a0a9746-8cca-11eb-2e39-ebb28670afa5
md"""

The function `constructu` estimates choice specific value functions and expected payoffs, $E_x[u(a_i,a_{-i},x)|a_i,x]$, given choice and transition probabilities. It does this by using the Hotz-Miller inversion (the relationship between choice probabilities and choice specific value functions), and then working with the Bellman-like equations.

To recover, $u(a_i,a_{-i},x)$ from the expected payoffs, we would impose a restriction on $u$ and use the relationship 

```math
E_x[u(a_i,a_{-i},x)|a_i,x] = \sum_{a_{-i}} \prod_{j\neq i} P(a_j|x) u(a_i, a_{-i}, x)
```
In this model, the restriction on $u$ is simply that $a_{-i}$ does not enter $u$ ($a_{-i}$ only affects future payoffs by shifting future $x$). In this case, we simply have $u(a_i, x) = E_x[u(a_i,a_{-i},x)|a_i,x]$

"""

# ╔═╡ 245e383e-8cce-11eb-01d8-755c79a03c59
sd = DG.simulate(g, 1000, out[2], burnin=0, x0=1);

# ╔═╡ 636335f4-8c24-11eb-2c1a-e5d5358f2af7
let
	Eu, _ = DGE.constructu(sd, g.β)
	pretty_table(String,
		DataFrame("x"=>statevec.(g.states),
				  "u"=>[g.u(1,[2,1],x) for x in g.states],
				  "û"=>Eu[1,2,:]),
		backend=:html) |> HTML
end

# ╔═╡ 2e53f28a-8cc3-11eb-1e2b-3bdb9288b364
md"""

## Inference

Since the estimator is relatively fast, using bootstrap for inference is reasonable. We must be careful to preserve the dependence in the data when resampling the data. Simply resampling observations with replacement would not be correct. If the data came from many independent markets, we could sample markets with replacement. However, the simulated data is from a single market for many time periods. We could use the block bootstrap to preserve the time dependence. An arguably better approach given the Markovian assumptions of the model, is the Markov bootstrap of Horowitz (19??).   

"""

# ╔═╡ bafacfec-8cdc-11eb-091c-e9ad41a9fc9a
let
	T = length(sd.x)
	B = 199
	N = g.N
	S = length(g.states)
	A = length(g.actions)	
	u = zeros(B, N, A, S)
	ubs = zeros(B, N, A, S)
	P = DGE.choiceprob(sd)
	Px = DGE.transition(sd)
	û,_ = DGE.constructu(sd,g.β)
	for b in 1:B
		sb = DG.simulate(g, T, out[2], burnin=0)
		u[B,:,:,:], _ = DGE.constructu(sb, g.β)		
		ubs[B,:,:,:],_ = DGE.constructu(DGE.markovbootstrap(sd,P,Px), g.β)
	end
	
	simse = sqrt.([var(u[:,2,2,x]) for x in g.states])
    bsse = sqrt.([var(ubs[:,2,2,x]) for x in g.states])
	pretty_table(String,
		DataFrame("x"=>statevec.(g.states),
				  "u(i=1,a=2,x)"=>[g.u(1,[2,2],x) for x in g.states],
				  "û(i=1,a=2,x)"=>û[1,2,:],
			      "bsSE"=>(s->"($(@sprintf("%.2f",s)))").(bsse),
				  "simSE"=>(s->"($(@sprintf("%.2f",s)))").(simse)),				  
		backend=:html) |> HTML
end
	

# ╔═╡ Cell order:
# ╟─dd4c95aa-8a82-11eb-3be1-9309fb5be5af
# ╟─596b18be-8a83-11eb-01fe-07833e8ac25e
# ╟─6a6bbfc4-8a83-11eb-308b-0d1d918f30f1
# ╠═b7f912b2-8b48-11eb-080e-c3edaf9b55c4
# ╟─8026965a-8b90-11eb-19aa-b17e1eff6034
# ╠═ed50e4f4-8a84-11eb-2e5b-075cfdea1280
# ╟─2451bac4-8b8d-11eb-0a68-53f3fd21ffad
# ╠═dfbf24ee-8a97-11eb-368f-555cd55c151e
# ╟─7f7ecd96-8b93-11eb-0505-0925707e55c2
# ╠═81dcea24-8b24-11eb-0c49-3bc494f41eb8
# ╟─3589d3b8-8b94-11eb-0351-9d70e67fc6ef
# ╠═4d99c88a-8b2f-11eb-12b0-47811a6bbed3
# ╟─83ad2cc2-8b94-11eb-2ef6-a59ccdb501f3
# ╠═4c7340f6-8b77-11eb-2f62-9f3f6422d95e
# ╟─a441b890-8b94-11eb-1817-c995add52fa8
# ╠═9a6e598c-8b7a-11eb-0b29-c57ecadec0aa
# ╟─1a253676-8beb-11eb-1d16-170c65346d9f
# ╠═348a21de-8beb-11eb-027f-5d519ea12572
# ╠═27ab599c-8c06-11eb-36c3-090611cb8458
# ╠═94110c2a-8c03-11eb-3d00-ebab8f1d3811
# ╟─0518cb98-8bec-11eb-1c55-e19103432968
# ╟─8dfbd0c6-8b75-11eb-3643-05334135f3ce
# ╠═89d97f3a-8be2-11eb-3faf-ad0a6dc22571
# ╠═5351f0d0-8be4-11eb-05da-75dfb199544f
# ╠═0af9d260-8be6-11eb-2d51-5f803d5456bf
# ╠═2732717a-8c07-11eb-3a00-d19cfae02477
# ╠═5b3a2268-8c07-11eb-31bb-a320fd9e7bb0
# ╟─647103a0-8b95-11eb-29bc-6d26c83d4a13
# ╠═93acf058-8c25-11eb-1cb5-d391f007119c
# ╟─1a0a9746-8cca-11eb-2e39-ebb28670afa5
# ╠═245e383e-8cce-11eb-01d8-755c79a03c59
# ╠═636335f4-8c24-11eb-2c1a-e5d5358f2af7
# ╟─2e53f28a-8cc3-11eb-1e2b-3bdb9288b364
# ╠═bafacfec-8cdc-11eb-091c-e9ad41a9fc9a
