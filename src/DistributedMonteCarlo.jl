#module DistributedMonteCarlo

using StaticArrays
using Distributed
import AltInplaceOpsInterface: add!, minus!, pow!, max!, min!

struct MonteCarloShot{DIM,MCT}
	coords::SVector{DIM,MCT}
	MonteCarloShot(coords::SVector{DIM,MCT}) where {DIM,MCT} = new{DIM,MCT}(coords)
end
coords(mcs::MonteCarloShot) = mcs.coords

struct MonteCarlo{DIM,MCT,RT}
	shots::Vector{MonteCarloShot{DIM,MCT}}
	n::Int
	tol::Float64
	rndF::Function
	results::Dict{String,RT}
	convergence_history::Dict{String,Vector{Float64}}
	function MonteCarlo(::Val{DIM},::Type{MCT},::Type{RT}, n, tol, rndF::F2) where {DIM,MCT,RT,F1<:Function,F2<:Function}
		MC = new{DIM,MCT,RT}(Vector{MonteCarloShot{DIM,MCT}}(undef,n),n,tol,rndF)#,Dict{String,RT}(),Dict{String,Float64}())
		for i = 1:MC.n
			#@info "$i/$(MC.n) Monte Carlo Shot"	
			ξs = SVector(MC.rndF()...)
			MC.shots[i] = MonteCarloShot(ξs)
		end
		return MC
	end
end


function distributed_𝔼(MC::MonteCarlo{DIM,MCT,RT}, fun::F, worker_ids::Vector{Int}) where {DIM,MCT,RT,F<:Function}
	
	sumthreadlock = Threads.Condition()
	wp = WorkerPool(worker_ids);
	num_workers = length(worker_ids)
	#results = RemoteChannel(()->Channel{RT}(length(worker_ids)+1));
	#intres = RemoteChannel(()->Channel{RT}(1));
	results = Channel{RT}(num_workers+1)
	intres = Channel{RT}(1)
	nresults = 0

	#@everywhere begin
	#	function do_job(fun,val,res)
	#		put!(res, fun(val))
	#		return 1
	#	end
	#end

	println("1")
	@async begin
		res = take!(results)
		println("first result")
		nresults += 1
		while nresults < MC.n
			_res = take!(results)
			nresults += 1
			add!(res,_res)
			println("result $nresults")
			sleep(0.001)		
		end
		put!(intres, res/nresults)
		println("done channel")
	end

	println("2")

	@sync begin
		for shot in MC.shots
			i = 0
			while !isready(wp)
				sleep(0.001)
				#println("sleep")
			end
			i += 1
			@async begin
				val = coords(shot)
				println(val)
				_fval = remotecall_fetch(fun, wp, val)				
				println("put")
				put!(results, _fval)
			end
			if i >= num_workers
				sleep(0.001)
				i = 0
			end
		end
	end
	println("3")
	return take!(intres)
end


#addprocs(80)
#
#@everywhere samplefun = x->begin; sleep(0.1); ones(Float64,20_000,20_000); end
#@everywhere begin
#	using StaticArrays
#end
#
#MC = MonteCarlo(Val(20),Float64,Matrix{Float64}, 10_000, 0.01, ()->randn(20))
#@time 𝔼 = distributed_𝔼(MC, samplefun, workers()) 
#
#MC = MonteCarlo(Val(20),Float64,Matrix{Float64}, 20_000, 0.01, ()->randn(20))
#@time 𝔼 = distributed_𝔼(MC, samplefun, workers()) 

#function start!(MC::MonteCarlo{DIM,MCT,RT}, worker_ids::Vector{Int}) where {DIM,MCT,RT}
#	i = 0
#	while i<=MC.n
#		@sync begin
#			for pid in worker_ids
#				i += 1
#				if i > MC.n
#					break
#				end
#				@info "$i/$(MC.n) Monte Carlo Shot"
#				ξs = MC.rndF()
#				mcs = MonteCarloShot(ξs,RT)
#				MC.shots[i] = mcs				
#				@async begin
#				    fval = remotecall_fetch(MC.Fun, pid, ξs)
#					set_val!(mcs,fval)
#				end
#			end
#		end
#	end
#end
#
#function restart!(MC::MonteCarlo{DIM,MCT,RT}, restartfunc::F, restartpathes, worker_ids::Vector{Int}) where {DIM,MCT,RT,F<:Function}
#	i = 0
#	n = length(restartpathes)
#	if length(MC.n) != n
#		resize!(MC.shots,n)
#	end
#	while i<=n
#		@sync begin
#			for pid in worker_ids
#				i += 1
#				if i > n
#					break
#				end
#				@info "$i/$(n) Monte Carlo Shot"
#				mcs = MonteCarloShot(SVector{DIM,MCT},RT)
#				MC.shots[i] = mcs
#				path = restartpathes[i]				
#				@async begin
#				    coords,fval = remotecall_fetch(restartfunc, pid, path)
#					set_coords!(mcs,coords)
#					set_val!(mcs,fval)
#				end
#			end
#		end
#	end
#end
#
#function continue!(MC::MonteCarlo{DIM,MCT,RT},n::Int,worker_ids::Vector{Int}) where {DIM,MCT,RT}
#	i = 0
#	while i<=n
#		@sync begin
#			for pid in worker_ids
#				i += 1
#				if i > n
#					break
#				end
#				@info "$(MC.n+i)/$(MC.n+n) Monte Carlo Shot"
#				ξs = MC.rndF()
#				mcs = MonteCarloShot(ξs,RT)
#				push!(MC.shots, mcs)			
#				@async begin
#				    fval = remotecall_fetch(MC.Fun, pid, ξs)
#					set_val!(mcs,fval)
#				end
#			end
#		end
#	end
#	MC.n += n
#	return nothing
#end
#
#function 𝔼(MC::MonteCarlo)
#	res = foldl(+,map(x->x.val,MC.shots),init=zero(MC.shots[1].val))
#	return res/MC.n
#end
#
#function var(MC::MonteCarlo)
#	_𝔼 = 𝔼(MC)
#	var = zero(_𝔼)
#	for i = 1:MC.n
#		var += (MC.shots[i].val - _𝔼) ^ 2.0
#	end
#	var /= (MC.n-1)
#	return var
#end

#end # module
