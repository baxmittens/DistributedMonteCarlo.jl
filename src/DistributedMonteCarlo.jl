module DistributedMonteCarlo

using StaticArrays
using Distributed
import AltInplaceOpsInterface: add!, minus!, pow!, max!, min!
using LinearAlgebra
using UnicodePlots
using LatinHypercubeSampling

mutable struct MonteCarloShot{DIM,MCT,RT}
	coords::SVector{DIM,MCT}
	res::Union{Nothing,RT}
	MonteCarloShot(coords::SVector{DIM,MCT},::Type{RT}) where {DIM,MCT,RT} = new{DIM,MCT,RT}(coords,nothing)
end
coords(mcs::MonteCarloShot) = mcs.coords

mutable struct MonteCarlo{DIM,MCT,RT}
	shots::Vector{MonteCarloShot{DIM,MCT,RT}}
	n::Int
	tol::Float64
	rndF::Function
	save_res::Bool
	convergence_history::Dict{String,Tuple{Vector{Float64},Vector{Float64}}}
	function MonteCarlo(::Val{DIM},::Type{MCT},::Type{RT}, n, tol, rndF::F2, save_res::Bool=false) where {DIM,MCT,RT,F2<:Function}
		MC = new{DIM,MCT,RT}(Vector{MonteCarloShot{DIM,MCT,RT}}(undef,n),n,tol,rndF,save_res,Dict{String,Tuple{Vector{Float64},Vector{Float64}}}())
		for i = 1:MC.n
			ξs = SVector(MC.rndF()...)
			MC.shots[i] = MonteCarloShot(ξs,RT)
		end
		return MC
	end
end

function load!(MC::MonteCarlo{DIM,MCT,RT}, restartpath) where {DIM,MCT,RT}
	snapshotdirs = readdir(restartpath)
	filter!(x->isdir(joinpath(restartpath,x)), snapshotdirs)
	n = length(snapshotdirs)
	if n > MC.n
		@warn "change size of n from $(MC.n) to $n"
		MC.n = n
		resize!(MC.shots,n)
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
			MC.shots[i] = MonteCarloShot(coords,RT)
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
		try
			res = deepcopy(take!(results))
			nresults += 1
			while nresults < MC.n
				_res = take!(results)
				nresults += 1
				add!(res,_res)
				if mod(nresults,1000) == 0
					println("n = $nresults")
				end
				#if mod(nresults, conv_interv) == 0
					#push!(conv_n, nresults)
					#push!(conv_norm, norm(res/nresults))
				#end
				sleep(0.0001)		
			end
			#if conv_n ∉ nresults
				#push!(conv_n, nresults)
				#push!(conv_norm, norm(res/nresults))
			#end
			rescopy = deepcopy(res)
			mul!(rescopy, 1.0/nresults)
			put!(intres, rescopy)
		catch e
			println(e)
		end
	end

	@sync begin
		for shot in MC.shots
			while !isready(wp) && length(results.data)<num_workers
				println("WorkerPool not ready")
				sleep(1)
			end
			@async begin
				val = coords(shot)
				_fval = remotecall_fetch(fun, wp, val, string(hash(val)))
				put!(results, _fval)
				if MC.save_res
					shot.res = _fval
				end
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
		res = deepcopy(take!(results))
		minus!(res,exp_val)
		#res .^= 2.0
		pow!(res, 2.0)
		nresults += 1
		while nresults < MC.n
			_res = deepcopy(take!(results))
			minus!(_res,exp_val)
			pow!(_res, 2.0)
			nresults += 1
			add!(res, _res)
			#if mod(nresults,1000) == 0
			#	println("n = $nresults")
			#end
			#if mod(nresults, conv_interv) == 0
			#	push!(conv_n, nresults)
			#	push!(conv_norm, norm(res/nresults))
			#end
			sleep(0.0001)		
		end
		#push!(conv_n, nresults)
		#push!(conv_norm, norm(res/nresults))
		rescopy = deepcopy(res)
		mul!(rescopy, 1.0/(nresults-1))
		put!(intres, rescopy)
	end

	@sync begin
		for shot in MC.shots
			while !isready(wp) && length(results.data)<num_workers
				sleep(0.0001)
			end
			@async begin
				val = coords(shot)
				#println(val)
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
	shotsA::Vector{MonteCarloShot{DIM,MCT,RT}}
	shotsB::Vector{MonteCarloShot{DIM,MCT,RT}}
	shotsA_B::Matrix{MonteCarloShot{DIM,MCT,RT}}
	n::Int
	tol::Float64
	rndF::Function
	convergence_history::Dict{String,Tuple{Vector{Float64},Vector{Float64}}}
	function MonteCarloSobol(::Val{DIM},::Type{MCT},::Type{RT}, n, tol, rndF::F2) where {DIM,MCT,RT,F2<:Function}
		shotsA = Vector{MonteCarloShot{DIM,MCT,RT}}(undef,n)
		shotsB = Vector{MonteCarloShot{DIM,MCT,RT}}(undef,n)
		shotsA_B = Matrix{MonteCarloShot{DIM,MCT,RT}}(undef,DIM,n)
		MC = new{DIM,MCT,RT}(shotsA,shotsB,shotsA_B,n,tol,rndF,Dict{String,Tuple{Vector{Float64},Vector{Float64}}}())
		for i = 1:MC.n
			ξs = SVector(MC.rndF()...)
			MC.shotsA[i] = MonteCarloShot(ξs,RT)
			ξs = SVector(MC.rndF()...)
			MC.shotsB[i] = MonteCarloShot(ξs,RT)
		end
		ξvec = zeros(MCT, DIM)
		for i = 1:DIM
			inds =  setdiff(1:DIM,i)
			for j = 1:MC.n
				ξvec[i] = MC.shotsB[j].coords[i]
				ξvec[inds] = MC.shotsA[j].coords[inds]				
				ξs = SVector(ξvec...)
				MC.shotsA_B[i,j] = MonteCarloShot(ξs,RT)
			end
		end
		return MC
	end
end

function load!(MC::MonteCarloSobol{DIM,MCT,RT}, restartpath) where {DIM,MCT,RT}
	snapshotdirsA = readdir(joinpath(restartpath,"A"))
	n = length(snapshotdirsA)
	if n > MC.n
		#@warn "change size of n from $(MC.n) to $n"
		#MC.n = n
		n = MC.n
		#resize!(MC.shotsA,n)
		#resize!(MC.shotsB,n)
	end
	_permA = sortperm(map(x->parse(Int,x),snapshotdirsA))
	for i = 1:n
		ind = _permA[i]
		#snapshotdir = readdir(joinpath(restartpath,"A",snapshotdirsA[i]))
		pars_txt = joinpath(restartpath,"A",snapshotdirsA[ind],"coords.txt")
		if isfile(pars_txt)
			f = open(pars_txt);
			lines = readlines(f)
			close(f)
			coords = SVector(map(x->parse(Float64,x),lines)...)
			MC.shotsA[i] = MonteCarloShot(coords,RT)
		end
	end
	snapshotdirsB = readdir(joinpath(restartpath,"B"))
	_permB = sortperm(map(x->parse(Int,x),snapshotdirsA))
	m = length(snapshotdirsB)
	if m > n
		m = n
	end
	for i = 1:m
		ind = _permB[i]
		#snapshotdir = readdir(joinpath(restartpath,"B",snapshotdirsB[i]))
		pars_txt = joinpath(restartpath,"B",snapshotdirsB[ind],"coords.txt")
		if isfile(pars_txt)
			f = open(pars_txt);
			lines = readlines(f)
			close(f)
			coords = SVector(map(x->parse(Float64,x),lines)...)
			MC.shotsB[i] = MonteCarloShot(coords,RT)
		end
	end
	shotsA_B = Matrix{MonteCarloShot{DIM,MCT,RT}}(undef,DIM,MC.n)
	ξvec = zeros(MCT, DIM)
	for i = 1:DIM
		inds =  setdiff(1:DIM,i)
		for j = 1:MC.n
			ξvec[i] = MC.shotsB[j].coords[i]
			ξvec[inds] = MC.shotsA[j].coords[inds]				
			ξs = SVector(ξvec...)
			shotsA_B[i,j] = MonteCarloShot(ξs,RT)
		end
	end
	MC.shotsA_B = shotsA_B
	return nothing
end

function distributed_sampling_A(MC::MonteCarloSobol{DIM,MCT,RT}, fun::F, worker_ids::Vector{Int}, verbose::Bool=false) where {DIM,MCT,RT,F<:Function}
	wp = WorkerPool(worker_ids);
	num_workers = length(worker_ids)
	results = Channel{RT}(num_workers+1)
	intres = Channel{RT}(1)
	nresults = 0

	#conv_n, conv_norm, conv_interv = Vector{Float64}(), Vector{Float64}(), floor(Int,MC.n/1000)
	conv_n, conv_norm, conv_interv = Vector{Float64}(), Vector{Float64}(), max(length(worker_ids),floor(Int,MC.n/1000))

	@async begin
		res = take!(results)
		nresults += 1		
		while nresults < MC.n
			_res = take!(results)
			nresults += 1
			add!(res,_res)
			if verbose && mod(nresults,1000) == 0
				println("n = $nresults of $(MC.n) total shots")
			end
			if mod(nresults, conv_interv) == 0
				push!(conv_n, nresults)
				push!(conv_norm, norm(res/nresults))
				if verbose
					println("convergence exp_val")
					display(scatterplot(conv_n,conv_norm))
				end
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
				#println("WorkerPool not ready")
				sleep(0.1)
			end
			@async begin
				val = coords(shot)
				#_fval = remotecall_fetch(fun, wp, val, string(hash(val)))
				_fval = remotecall_fetch(fun, wp, val, joinpath("A",string(numshot)))
				put!(results, _fval)
			end
			sleep(0.0001)
		end
	end
	MC.convergence_history["exp_val"] = (conv_n, conv_norm)
	return take!(intres)
end

function distributed_sampling_B(MC::MonteCarloSobol{DIM,MCT,RT}, exp_val::RT, fun::F, worker_ids::Vector{Int}, verbose::Bool=false) where {DIM,MCT,RT,F<:Function}
	wp = WorkerPool(worker_ids);
	num_workers = length(worker_ids)
	results = Channel{RT}(num_workers+1)
	intres = Channel{RT}(1)
	nresults = 0

	#conv_n, conv_norm, conv_interv = Vector{Float64}(), Vector{Float64}(), floor(Int,MC.n/1000)
	conv_n, conv_norm, conv_interv = Vector{Float64}(), Vector{Float64}(), max(length(worker_ids),floor(Int,MC.n/1000))

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

			if verbose && mod(nresults,1000) == 0
				println("n = $nresults of $(MC.n) total shots")
			end
			if mod(nresults, conv_interv) == 0
				push!(conv_n, nresults)
				push!(conv_norm, norm(res/nresults))
				if verbose
					println("convergence var_val")
					display(scatterplot(conv_n,conv_norm))				
				end
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
				#println("WorkerPool not ready")
				sleep(0.1)
			end
			@async begin
				val = coords(shot)
				#_fval = remotecall_fetch(fun, wp, val, string(hash(val)))
				_fval = remotecall_fetch(fun, wp, val, joinpath("B",string(numshot)))
				put!(results, _fval)
			end
			sleep(0.0001)
		end
	end
	MC.convergence_history["var_val"] = (conv_n, conv_norm)
	return take!(intres)
end

function distributed_sampling_A_B(MC::MonteCarloSobol{DIM,MCT,RT}, fun::F, worker_ids::Vector{Int}, verbose::Bool=false) where {DIM,MCT,RT,F<:Function}
	wp = WorkerPool(worker_ids);
	num_workers = length(worker_ids)
	results = Channel{Tuple{RT,RT,Int}}(num_workers+1)
	intres = Channel{Tuple{Vector{RT},Vector{RT}}}(1)
	nresults = 0
	nresults_i = zeros(Int,DIM)

	conv_n_i, conv_norm_i, conv_interv = Vector{Vector{Float64}}(undef,DIM), Vector{Vector{Float64}}(undef,DIM), max(length(worker_ids),floor(Int,MC.n/1000))
	conv_rel_norm_i = Vector{Vector{Float64}}(undef,DIM)
	
	for i in 1:DIM
		conv_n_i[i] = Vector{Float64}()
		conv_norm_i[i] = Vector{Float64}()
		conv_rel_norm_i[i] = Vector{Float64}()
	end

	restmp = Vector{RT}(undef,DIM)
	restmp_totvar = Vector{RT}(undef,DIM)

	@async begin
		try	
			while nresults < MC.n*DIM
				res,restot,resi = take!(results)
				nresults += 1
				nresults_i[resi] += 1
				if isassigned(restmp,resi)
					add!(restmp[resi],res)
					add!(restmp_totvar[resi],restot)
					#println(resi," ",res)
					#println(restmp[resi])
					#println()
				else
					restmp[resi] = res
					restmp_totvar[resi] = restot
				end
				if verbose && mod(nresults,1000) == 0
					println("n = $nresults of $(MC.n*DIM) total shots")
				end
				if mod(nresults_i[resi], conv_interv) == 0
					push!(conv_n_i[resi], nresults_i[resi])
					push!(conv_norm_i[resi], norm(restmp[resi])/nresults_i[resi])
					if verbose
						println("convergence S_$resi")
						display(scatterplot(conv_n_i[resi],conv_norm_i[resi]))
					end
					if length(conv_n_i[resi]) > 1
						conv_act_i = length(conv_n_i[resi])						
						push!(conv_rel_norm_i[resi], abs(conv_norm_i[resi][conv_act_i]-conv_norm_i[resi][conv_act_i-1])/conv_norm_i[resi][1])
						if verbose
							println("convergence rel S_$resi")
							display(scatterplot(conv_n_i[resi][2:end],conv_rel_norm_i[resi]))
						end
					end	
				end
				sleep(0.0001)		
			end
			for resi = 1:DIM
				#println("divide $resi by $(nresults_i[resi])")
				mul!(restmp[resi], 1.0/nresults_i[resi])
				mul!(restmp_totvar[resi], 1.0/(2*nresults_i[resi]))
			end
			put!(intres, (restmp, restmp_totvar))
		catch e
			println(e)
			rethrow(e)
		end
	end

	#lin_inds = LinearIndices(size(MC.shotsA_B))
	@sync begin
		for num_i in 1:size(MC.shotsA_B,1)
			for num_j in 1:size(MC.shotsA_B,2)
				shotA_B = MC.shotsA_B[num_i,num_j]
				while !isready(wp) && length(results.data)<num_workers
					#println("WorkerPool not ready")
					sleep(0.1)
				end

				@async begin
					valA_B = coords(shotA_B)					
					valA = coords(MC.shotsA[num_j])					
					valB = coords(MC.shotsB[num_j])
					ID = string(num_i)*"_"*string(num_j)								
					#resA_B = remotecall_fetch(fun, wp, valA_B, joinpath(MC.restartpath,"A_B",string(lin_inds[num_i,num_j])))
					resA_B = remotecall_fetch(fun, wp, valA_B, joinpath("A_B",ID))
					resA = remotecall_fetch(fun, wp, valA, joinpath("A",string(num_j)))
					resB = remotecall_fetch(fun, wp, valB, joinpath("B",string(num_j)))
					
					copy_resA_B = deepcopy(resA_B)
					
					minus!(resA_B,resA)
					mul!(resA_B,resB)

					minus!(resA,copy_resA_B)
					pow!(resA,2.0)

					put!(results, (resA_B,resA,num_i))
					#put!(results, (resA_B,zeros(length(resA_B)),num_i))
				end
				sleep(0.0001)
			end
		end
	end

	retval = take!(intres)

	for i = 1:DIM
		MC.convergence_history["S_$i"] = (conv_n_i[i], conv_norm_i[i])
		MC.convergence_history["relS_$i"] = (conv_n_i[i][2:end], conv_rel_norm_i[i])
	end

	return retval
end

function distributed_Sobol_Vars(MC::MonteCarloSobol{DIM,MCT,RT}, fun::F, worker_ids::Vector{Int}, verbose::Bool=false) where {DIM,MCT,RT,F<:Function}
	println("Distributed sampling A")
	@time expval = DistributedMonteCarlo.distributed_sampling_A(MC, fun, worker_ids, verbose)
	println("Distributed sampling B")
	@time varval = DistributedMonteCarlo.distributed_sampling_B(MC, expval, fun, worker_ids, verbose)
	println("Distributed sampling A_B")
	@time sobolvars,totsobolvars = DistributedMonteCarlo.distributed_sampling_A_B(MC, fun, worker_ids, verbose)
	return expval, varval, sobolvars, totsobolvars
end

mutable struct MorrisTrajectory{DIM,MT,RT}
    point::SVector{DIM,MT}
    traj::Vector{SVector{DIM,MT}}
    Δ::MT
end

function MorrisTrajectory(::Type{Val{DIM}}, ::Type{MT}, ::Type{RT}, rndF::F) where {DIM, MT, RT, F<:Function}
    point = SVector(rndF()...)
    Δmax = minimum(1.0 .- point)
    Δ = rand(MT)*Δmax
    traj = Vector{SVector{DIM,MT}}()
    for i = 1:DIM
    	_ei = zeros(MT,DIM)
        _ei[i] = Δ
        ei = SVector(_ei...)
        Δx = point.+ei
    	push!(traj, Δx)
    end
    return MorrisTrajectory{DIM,MT,RT}(point, traj, Δ)
end

function MorrisTrajectory(::Type{Val{DIM}}, ::Type{MT}, ::Type{RT}, _point::AbstractVector{MT}) where {DIM, MT, RT}
    point = SVector{DIM}(_point...)
    Δmax = minimum(1.0 .- point)
    Δ = rand(MT)*Δmax
    traj = Vector{SVector{DIM,MT}}()
    for i = 1:DIM
    	_ei = zeros(MT,DIM)
        _ei[i] = Δ
        ei = SVector(_ei...)
        Δx = point.+ei
    	push!(traj, Δx)
    end
    return MorrisTrajectory{DIM,MT,RT}(point, traj, Δ)
end

mutable struct MonteCarloMorris{DIM,MT,RT}
    trajectories::Vector{MorrisTrajectory{DIM,MT,RT}}
    n_trajectories::Int
    rndF::Function
	convergence_history::Dict{String,Tuple{Vector{Float64},Vector{Float64}}}
end

function lhs_sampling!(mcm::MonteCarloMorris{DIM,MT,RT}, Δmin_to_boarder=0.05) where {DIM, MT, RT}	
	plan, _ = LHCoptim(mcm.n_trajectories,DIM,2*DIM)
	scaled_plan = scaleLHC(plan,[(-one(MT)+Δmin_to_boarder,one(MT)-Δmin_to_boarder) for i in 1:DIM])
	for i in 1:mcm.n_trajectories
		point = scaled_plan[i,:]
		mcm.trajectories[i] = MorrisTrajectory(Val{DIM}, MT, RT, point)
    end
	return nothing
end

function MonteCarloMorris(::Val{DIM}, ::Type{MT}, ::Type{RT}, n_trajectories, rndF::F) where {DIM, MT, RT, F<:Function}
    trajectories = Vector{MorrisTrajectory{DIM,MT,RT}}(undef, n_trajectories)
    conv_hist = Dict{String,Tuple{Vector{Float64},Vector{Float64}}}()
    for i in 1:n_trajectories
        trajectories[i] = MorrisTrajectory(Val{DIM}, MT, RT, rndF)
    end
    mcm = MonteCarloMorris{DIM,MT,RT}(trajectories, n_trajectories, rndF, conv_hist)
    return mcm
end

function load!(MC::MonteCarloMorris{DIM,MT,RT}, restartpath) where {DIM,MT,RT}
	snapshotdirs = readdir(restartpath)
	traj_root_points_dirs = filter(x->!contains(x,"_"),snapshotdirs)
	traj_sortperm = sortperm(map(x->parse(Int,x),traj_root_points_dirs))
	traj_root_points_dirs_sorted = traj_root_points_dirs[traj_sortperm]
	n = length(traj_root_points_dirs_sorted)
	for i = 1:min(n,MC.n_trajectories)
		pars_txt = joinpath(restartpath,traj_root_points_dirs_sorted[i],"coords.txt")
		Δ = 0.0
		if isfile(pars_txt)
			f = open(pars_txt);
			lines = readlines(f)
			close(f)
			coords = SVector(map(x->parse(Float64,x),lines)...)
			MC.trajectories[i].point = coords
		end
		for j = 1:DIM
			traj_dir_j = joinpath(restartpath,traj_root_points_dirs_sorted[i]*"_"*"$j")
			pars_txt = joinpath(traj_dir_j,"coords.txt")
			if isfile(pars_txt)
				f = open(pars_txt);
				lines = readlines(f)
				close(f)
				coords = SVector(map(x->parse(Float64,x),lines)...)
				MC.trajectories[i].traj[j] = coords
			else
				println(traj_dir_j)
				println(pars_txt)
				error("somethings wrong. reset stochastic model")
			end
		end
		Δ = MC.trajectories[i].traj[1][1]-MC.trajectories[i].point[1]
		MC.trajectories[i].Δ = Δ
	end
	return nothing
end

function distributed_means(MC::MonteCarloMorris{DIM,MT,RT}, fun::F, worker_ids::Vector{Int}) where {DIM,MT,RT,F<:Function}
	
	sumthreadlock = Threads.Condition()
	wp = WorkerPool(worker_ids);
	num_workers = length(worker_ids)
	results = Channel{Vector{RT}}(num_workers+1)
	intres = Channel{Tuple{Vector{RT},Vector{RT}}}(1)
	nresults = 0

	conv_n_i, conv_norm_i, conv_interv = Vector{Vector{Float64}}(undef,DIM), Vector{Vector{Float64}}(undef,DIM), max(length(worker_ids),floor(Int,MC.n_trajectories/1000))
	conv_rel_norm_i = Vector{Vector{Float64}}(undef,DIM)
	
	for i in 1:DIM
		conv_n_i[i] = Vector{Float64}()
		conv_norm_i[i] = Vector{Float64}()
		conv_rel_norm_i[i] = Vector{Float64}()
	end

	@async begin
		ees = take!(results)
		ees_abs = [deepcopy(ee) for ee in ees]
		for i in 1:DIM
			pow!(ees_abs[i],2.0)
			pow!(ees_abs[i],0.5)
		end
		nresults += 1
		while nresults < MC.n_trajectories
			ees_i = take!(results)
			nresults += 1
			for i in 1:DIM
				add!(ees[i],ees_i[i])
			end
			for i in 1:DIM
				pow!(ees_i[i],2.0)
				pow!(ees_i[i],0.5)
				add!(ees_abs[i],ees_i[i])
			end
			#if mod(nresults,1000) == 0
			#	println("n = $nresults")
			#end
			if mod(nresults, conv_interv) == 0
				for resi in 1:DIM
					push!(conv_n_i[resi], nresults)
					push!(conv_norm_i[resi], norm(ees_abs[resi])/nresults)
					if false
						println("convergence S_$resi")
						display(scatterplot(conv_n_i[resi],conv_norm_i[resi]))
					end
					if length(conv_n_i[resi]) > 1
						conv_act_i = length(conv_n_i[resi])						
						push!(conv_rel_norm_i[resi], abs(conv_norm_i[resi][conv_act_i]-conv_norm_i[resi][conv_act_i-1])/conv_norm_i[resi][1])
						if false
							println("convergence rel S_$resi")
							display(scatterplot(conv_n_i[resi][2:end],conv_rel_norm_i[resi]))
						end
					end
				end	
			end
			sleep(0.0001)		
		end
		#if conv_n ∉ nresults
		#	push!(conv_n, nresults)
		#	push!(conv_norm, norm(res/nresults))
		#end
		for i in 1:DIM
			mul!(ees[i],1/nresults)
			mul!(ees_abs[i],1/nresults)
		end
		put!(intres, (ees,ees_abs))
	end

	@sync begin
		for (i,traj) in enumerate(MC.trajectories)
			while !isready(wp) && length(results.data)<num_workers
				println("WorkerPool not ready")
				sleep(1)
			end
			@async begin
				val = traj.point
				#println(val)
				_fval = remotecall_fetch(fun, wp, val, string(i))
				ees = [remotecall_fetch(fun, wp, traj.traj[j], string(i)*"_"*string(j)) for j in 1:length(traj.traj)]
				for i = 1:DIM
					minus!(ees[i],_fval)
					mul!(ees[i],1/traj.Δ)
				end
				put!(results, ees)
			end
			sleep(0.0001)
		end
	end

	retval = take!(intres)

	for i = 1:DIM
		MC.convergence_history["S_$i"] = (conv_n_i[i], conv_norm_i[i])
		MC.convergence_history["relS_$i"] = (conv_n_i[i][2:end], conv_rel_norm_i[i])
	end

	return retval
end

export MonteCarlo, MonteCarloShot, load!, distributed_𝔼, distributed_var, MonteCarloSobol, MonteCarloMorris

end #module
