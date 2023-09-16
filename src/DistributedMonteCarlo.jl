module DistributedMonteCarlo

using StaticArrays
using Distributed
import AltInplaceOpsInterface: add!, minus!, pow!, max!, min!
using LinearAlgebra

struct MonteCarloShot{DIM,MCT}
	coords::SVector{DIM,MCT}
	MonteCarloShot(coords::SVector{DIM,MCT}) where {DIM,MCT} = new{DIM,MCT}(coords)
end
coords(mcs::MonteCarloShot) = mcs.coords

mutable struct MonteCarlo{DIM,MCT,RT}
	shots::Vector{MonteCarloShot{DIM,MCT}}
	n::Int
	tol::Float64
	rndF::Function
	convergence_history::Dict{String,Tuple{Vector{Float64},Vector{Float64}}}
	function MonteCarlo(::Val{DIM},::Type{MCT},::Type{RT}, n, tol, rndF::F2) where {DIM,MCT,RT,F2<:Function}
		MC = new{DIM,MCT,RT}(Vector{MonteCarloShot{DIM,MCT}}(undef,n),n,tol,rndF,Dict{String,Tuple{Vector{Float64},Vector{Float64}}}())
		for i = 1:MC.n
			ξs = SVector(MC.rndF()...)
			MC.shots[i] = MonteCarloShot(ξs)
		end
		return MC
	end
end

function load!(MC::MonteCarlo{DIM,MCT,RT}, restartpath) where {DIM,MCT,RT}
	snapshotdirs = readdir(restartpath)
	n = length(snapshotdirs)
	if n > MC.n
		@warn "change size of n from $(MC.n) to $n"
		MC.n = n
	end
	for i = 1:MC.n
		snapshotdir = readdir(joinpath(restartpath,snapshotdirs[i]))
		pars_txt = joinpath(restartpath,snapshotdirs[i],"coords.txt")
		if isfile(pars_txt)
			f = open(pars_txt);
			lines = readlines(f)
			close(f)
			coords = SVector(map(x->parse(Float64,x),lines)...)
			@assert snapshotdirs[i]==string(hash(coords))
			MC.shots[i] = MonteCarloShot(coords)
		end
	end
	return nothing
end

function distributed_𝔼(MC::MonteCarlo{DIM,MCT,RT}, fun::F, worker_ids::Vector{Int}) where {DIM,MCT,RT,F<:Function}
	
	sumthreadlock = Threads.Condition()
	wp = WorkerPool(worker_ids);
	num_workers = length(worker_ids)
	results = Channel{RT}(num_workers+1)
	intres = Channel{RT}(1)
	nresults = 0

	conv_n, conv_norm, conv_interv = Vector{Float64}(), Vector{Float64}(), floor(Int,MC.n/100)

	@async begin
		res = take!(results)
		nresults += 1
		while nresults < MC.n
			_res = take!(results)
			nresults += 1
			add!(res,_res)
			if mod(nresults,1000) == 0
				println("n = $nresults")
			end
			if mod(nresults, conv_interv) == 0
				push!(conv_n, nresults)
				push!(conv_norm, norm(res/nresults))
			end
			sleep(0.001)		
		end
		push!(conv_n, nresults)
		push!(conv_norm, norm(res/nresults))
		put!(intres, res/nresults)
	end

	@sync begin
		for shot in MC.shots
			i = 0
			while !isready(wp)
				sleep(0.001)
			end
			i += 1
			@async begin
				val = coords(shot)
				println(val)
				_fval = remotecall_fetch(fun, wp, val, string(hash(val)))
				put!(results, _fval)
			end
			if i >= num_workers
				sleep(0.001)
				i = 0
			end
		end
	end
	MC.convergence_history["exp_val"] = (conv_n, conv_norm)
	return take!(intres)
end

function distributed_var(MC::MonteCarlo{DIM,MCT,RT}, fun::F, exp_val::RT, worker_ids::Vector{Int}) where {DIM,MCT,RT,F<:Function}
	
	sumthreadlock = Threads.Condition()
	wp = WorkerPool(worker_ids);
	num_workers = length(worker_ids)
	results = Channel{RT}(num_workers+1)
	intres = Channel{RT}(1)
	nresults = 0

	conv_n, conv_norm, conv_interv = Vector{Float64}(), Vector{Float64}(), floor(Int,MC.n/100)

	@async begin
		res = take!(results)
		nresults += 1
		while nresults < MC.n
			_res = take!(results)
			nresults += 1
			add!(res,(_res - exp_val)^2.0)
			if mod(nresults,1000) == 0
				println("n = $nresults")
			end
			if mod(nresults, conv_interv) == 0
				push!(conv_n, nresults)
				push!(conv_norm, norm(res/nresults))
			end
			sleep(0.001)		
		end
		push!(conv_n, nresults)
		push!(conv_norm, norm(res/nresults))
		put!(intres, res/(nresults-1))
	end

	@sync begin
		for shot in MC.shots
			i = 0
			while !isready(wp)
				sleep(0.001)
			end
			i += 1
			@async begin
				val = coords(shot)
				println(val)
				_fval = remotecall_fetch(fun, wp, val, string(hash(val)))
				put!(results, _fval)
			end
			if i >= num_workers
				sleep(0.001)
				i = 0
			end
		end
	end

	MC.convergence_history["var_val"] = (conv_n, conv_norm)
	return take!(intres)
end

end #module
