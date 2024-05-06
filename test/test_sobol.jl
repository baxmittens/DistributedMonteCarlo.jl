using DistributedMonteCarlo
using Distributed

addprocs(80)

@everywhere begin

import AltInplaceOpsInterface
import AltInplaceOpsInterface: add!, minus!, pow!
import LinearAlgebra: mul!
using StaticArrays

AltInplaceOpsInterface.add!(a::Vector{Float64}, b::Vector{Float64}) = a .+= b
mul!(a::Vector{Float64}, b::Float64) = a .*= b
mul!(a::Vector{Float64}, b::Vector{Float64}) = a .*= b
mul!(a::Vector{Float64}, b::Vector{Float64}, c::Float64) = begin; for i = 1:length(a); a[i] = b[i]*c; end; return nothing; end
AltInplaceOpsInterface.minus!(a::Vector{Float64}, b::Vector{Float64}) = a .-= b
AltInplaceOpsInterface.pow!(a::Vector{Float64}, b::Int64) = a .^= b
AltInplaceOpsInterface.pow!(a::Vector{Float64}, b::Float64) = a .^= b

function func(xx)
	x,y,z = xx
	return [x,x+y,z*x^2]
end
func(xx::SVector{3, Float64}, ::String) = func(xx)
end

DIM = 3
valdim = Val(DIM)
nshots = 10_000
MC = MonteCarloSobol(valdim,Float64, Vector{Float64}, nshots, 0.01, ()->rand(DIM).*2.0.-1.0)
#expval = DistributedMonteCarlo.distributed_sampling_A(MC, func, workers())
#varval = DistributedMonteCarlo.distributed_sampling_B(MC, expval, func, workers())
#sobolinds = DistributedMonteCarlo.distributed_sampling_A_B(MC, func, workers())
expval, varval, sobolvars = DistributedMonteCarlo.distributed_Sobol_Vars(MC, func, workers())
