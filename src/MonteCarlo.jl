using StaticArrays

mutable struct MonteCarloShot{DIM,MCT,RT}
	coords::SVector{DIM,MCT}
	val::RT
	MonteCarloShot(::Type{SVector{DIM,MCT}},::Type{RT}) where {DIM,MCT,RT} = new{DIM,MCT,RT}() #incomplete constructor
	MonteCarloShot(coords::SVector{DIM,MCT},::Type{RT}) where {DIM,MCT,RT} = new{DIM,MCT,RT}(coords) #incomplete constructor
	MonteCarloShot(coords::SVector{DIM,MCT},val::RT) where {DIM,MCT,RT} = new{DIM,MCT,RT}(coords,val)
end
set_coords!(mcs::MonteCarloShot{DIM,MCT,RT}, coords::SVector{DIM,MCT}) where {DIM,MCT,RT} = mcs.coords=coords
set_val!(mcs::MonteCarloShot{DIM,MCT,RT}, val::RT) where {DIM,MCT,RT} = mcs.val=val

mutable struct MonteCarlo{DIM,MCT,RT}
	shots::Vector{MonteCarloShot{DIM,MCT,RT}}
	n::Int
	tol::Float64
	Fun::Function
	rndF::Function #randf() -> SVector{DIM,MCT} 
	function MonteCarlo(::Val{DIM},::Type{MCT},::Type{RT}, n, tol, Fun::F1, rndF::F2) where {DIM,MCT,RT,F1<:Function,F2<:Function}
		new{DIM,MCT,RT}(Vector{MonteCarloShot{DIM,MCT,RT}}(undef,n),n,tol,Fun,rndF)
	end
end

function start!(MC::MonteCarlo)
	for i = 1:MC.n
		@info "$i/$(MC.n) Monte Carlo Shot"	
		ξs = MC.rndF()
		res = MC.Fun(ξs)
		MC.shots[i] = MonteCarloShot(ξs,res)
	end
end

function start!(MC::MonteCarlo{DIM,MCT,RT}, worker_ids::Vector{Int}) where {DIM,MCT,RT}
	i = 0
	while i<=MC.n
		@sync begin
			for pid in worker_ids
				i += 1
				if i > MC.n
					break
				end
				@info "$i/$(MC.n) Monte Carlo Shot"
				ξs = MC.rndF()
				mcs = MonteCarloShot(ξs,RT)
				MC.shots[i] = mcs				
				@async begin
				    fval = remotecall_fetch(MC.Fun, pid, ξs)
					set_val!(mcs,fval)
				end
			end
		end
	end
end

function restart!(MC::MonteCarlo{DIM,MCT,RT}, restartfunc::F, restartpathes, worker_ids::Vector{Int}) where {DIM,MCT,RT,F<:Function}
	i = 0
	n = length(restartpathes)
	if length(MC.n) != n
		resize!(MC.shots,n)
	end
	while i<=n
		@sync begin
			for pid in worker_ids
				i += 1
				if i > n
					break
				end
				@info "$i/$(n) Monte Carlo Shot"
				mcs = MonteCarloShot(SVector{DIM,MCT},RT)
				MC.shots[i] = mcs
				path = restartpathes[i]				
				@async begin
				    coords,fval = remotecall_fetch(restartfunc, pid, path)
					set_coords!(mcs,coords)
					set_val!(mcs,fval)
				end
			end
		end
	end
end


function continue!(MC::MonteCarlo{DIM,MCT,RT},n::Int,worker_ids::Vector{Int}) where {DIM,MCT,RT}
	i = 0
	while i<=n
		@sync begin
			for pid in worker_ids
				i += 1
				if i > n
					break
				end
				@info "$(MC.n+i)/$(MC.n+n) Monte Carlo Shot"
				ξs = MC.rndF()
				mcs = MonteCarloShot(ξs,RT)
				push!(MC.shots, mcs)			
				@async begin
				    fval = remotecall_fetch(MC.Fun, pid, ξs)
					set_val!(mcs,fval)
				end
			end
		end
	end
	MC.n += n
	return nothing
end

function 𝔼(MC::MonteCarlo)
	res = foldl(+,map(x->x.val,MC.shots),init=zero(MC.shots[1].val))
	return res/MC.n
end

function var(MC::MonteCarlo)
	_𝔼 = 𝔼(MC)
	var = zero(_𝔼)
	for i = 1:MC.n
		var += (MC.shots[i].val - _𝔼) ^ 2.0
	end
	var /= (MC.n-1)
	return var
end