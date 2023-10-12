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
	for i = 1:n
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
			sleep(0.0001)		
		end
		if conv_n ∉ nresults
			push!(conv_n, nresults)
			push!(conv_norm, norm(res/nresults))
		end
		put!(intres, res/nresults)
	end

	@sync begin
		for shot in MC.shots
			while !isready(wp) && length(results.data)<num_workers
				println("WorkerPool not ready")
				sleep(1)
			end
			@async begin
				val = coords(shot)
				println(val)
				_fval = remotecall_fetch(fun, wp, val, string(hash(val)))
				put!(results, _fval)
			end
			sleep(0.0001)
		end
	end
	MC.convergence_history["exp_val"] = (conv_n, conv_norm)
	return take!(intres)
end

function distributed_var(MC::MonteCarlo{DIM,MCT,RT}, fun::F, exp_val::RT, worker_ids::Vector{Int}) where {DIM,MCT,RT,F<:Function}
	
	sumthreadlock = Threads.Condition()
	wp = WorkerPool(worker_ids);
	num_workers = length(worker_ids)
	results = Channel{RT}(2*num_workers)
	intres = Channel{RT}(1)
	nresults = 0

	conv_n, conv_norm, conv_interv = Vector{Float64}(), Vector{Float64}(), floor(Int,MC.n/100)

	@async begin
		res = take!(results)
		minus!(res,exp_val)
		res .^= 2.0
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
			sleep(0.0001)		
		end
		push!(conv_n, nresults)
		push!(conv_norm, norm(res/nresults))
		put!(intres, res/(nresults-1))
	end

	@sync begin
		for shot in MC.shots
			while !isready(wp) && length(results.data)<num_workers
				sleep(0.0001)
			end
			@async begin
				val = coords(shot)
				println(val)
				_fval = remotecall_fetch(fun, wp, val, string(hash(val)))
				put!(results, _fval)
			end
			sleep(0.0001)
		end
	end

	MC.convergence_history["var_val"] = (conv_n, conv_norm)
	return take!(intres)
end

mutable struct MonteCarloSobol{DIM,MCT,RT}
	shotsA::Vector{MonteCarloShot{DIM,MCT}}
	shotsB::Vector{MonteCarloShot{DIM,MCT}}
	shotsA_B::Matrix{MonteCarloShot{DIM,MCT}}
	restartpath::String
	n::Int
	tol::Float64
	rndF::Function
	convergence_history::Dict{String,Tuple{Vector{Float64},Vector{Float64}}}
	function MonteCarloSobol(::Val{DIM},::Type{MCT},::Type{RT}, n, tol, rndF::F2, restartpath="./") where {DIM,MCT,RT,F2<:Function}
		shotsA = Vector{MonteCarloShot{DIM,MCT}}(undef,n)
		shotsB = Vector{MonteCarloShot{DIM,MCT}}(undef,n)
		shotsA_B = Matrix{MonteCarloShot{DIM,MCT}}(undef,DIM,n)
		MC = new{DIM,MCT,RT}(shotsA,shotsB,shotsA_B,restartpath,n,tol,rndF,Dict{String,Tuple{Vector{Float64},Vector{Float64}}}())
		for i = 1:MC.n
			ξs = SVector(MC.rndF()...)
			MC.shotsA[i] = MonteCarloShot(ξs)
			ξs = SVector(MC.rndF()...)
			MC.shotsB[i] = MonteCarloShot(ξs)
		end
		ξvec = zeros(MCT, DIM)
		for i = 1:DIM
			inds =  setdiff(1:DIM,i)
			for j = 1:MC.n
				ξvec[i] = MC.shotsB[j].coords[i]
				ξvec[inds] = MC.shotsA[j].coords[inds]				
				ξs = SVector(ξvec...)
				MC.shotsA_B[i,j] = MonteCarloShot(ξs)
			end
		end
		return MC
	end
end

function distributed_sampling_A(MC::MonteCarloSobol{DIM,MCT,RT}, fun::F, worker_ids::Vector{Int}) where {DIM,MCT,RT,F<:Function}
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
			sleep(0.0001)		
		end
		if conv_n ∉ nresults
			push!(conv_n, nresults)
			push!(conv_norm, norm(res/nresults))
		end
		put!(intres, res/nresults)
	end

	@sync begin
		for (numshot,shot) in enumerate(MC.shotsA)
			while !isready(wp) && length(results.data)<num_workers
				println("WorkerPool not ready")
				sleep(1)
			end
			@async begin
				val = coords(shot)
				#_fval = remotecall_fetch(fun, wp, val, string(hash(val)))
				_fval = remotecall_fetch(fun, wp, val, jointpath(restartpath,string(numshot)))
				put!(results, _fval)
			end
			sleep(0.0001)
		end
	end
	MC.convergence_history["exp_val"] = (conv_n, conv_norm)
	return take!(intres)
end

function distributed_sampling_B(MC::MonteCarloSobol{DIM,MCT,RT}, exp_val::RT, fun::F, worker_ids::Vector{Int}) where {DIM,MCT,RT,F<:Function}
	wp = WorkerPool(worker_ids);
	num_workers = length(worker_ids)
	results = Channel{RT}(num_workers+1)
	intres = Channel{RT}(1)
	nresults = 0

	conv_n, conv_norm, conv_interv = Vector{Float64}(), Vector{Float64}(), floor(Int,MC.n/100)

	@async begin
		res = take!(results)
		tmp = similar(res)
		
		fill!(tmp,0.0)
		add!(tmp,res)
		minus!(tmp,exp_val)
		pow!(tmp,2.0)
		fill!(res,0.0)
		add!(res,tmp)

		nresults += 1		
		while nresults < MC.n
			_res = take!(results)
			nresults += 1
			
			fill!(tmp,0.0)
			add!(tmp,_res)
			minus!(tmp,exp_val)
			pow!(tmp,2.0)
			add!(res,tmp)

			if mod(nresults,1000) == 0
				println("n = $nresults")
			end
			if mod(nresults, conv_interv) == 0
				push!(conv_n, nresults)
				push!(conv_norm, norm(res/nresults))
			end
			sleep(0.0001)		
		end
		if conv_n ∉ nresults
			push!(conv_n, nresults)
			push!(conv_norm, norm(res/nresults))
		end
		put!(intres, res/(nresults-1))
	end

	@sync begin
		for (numshot,shot) in enumerate(MC.shotsB)
			while !isready(wp) && length(results.data)<num_workers
				println("WorkerPool not ready")
				sleep(1)
			end
			@async begin
				val = coords(shot)
				#_fval = remotecall_fetch(fun, wp, val, string(hash(val)))
				_fval = remotecall_fetch(fun, wp, val, jointpath(restartpath,string(numshot)))
				put!(results, _fval)
			end
			sleep(0.0001)
		end
	end
	MC.convergence_history["exp_val"] = (conv_n, conv_norm)
	return take!(intres)
end

export MonteCarlo, MonteCarloShot, load!, distributed_𝔼, distributed_var, MonteCarloSobol

end #module
