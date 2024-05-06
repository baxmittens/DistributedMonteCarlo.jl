# DistributedMonteCarlo
A Monte Carlo implementation in the julia language using [Distributed.jl](https://docs.julialang.org/en/v1/stdlib/Distributed/) and [AltInplaceOpsInterface.jl](https://github.com/baxmittens/AltInplaceOpsInterface.jl)

This code is rather slow. Its purpose is to be able to perform Monte Carlo analyses that, at one time, would not fit in main memory or where the snapshots need to be computed on distributed infrastructures. 

[AltInplaceOpsInterface.jl](https://github.com/baxmittens/AltInplaceOpsInterface.jl) provides in-place operation for the Monte Carlo return type and is used due to compatibility to old julia code and could be replaced by implementing the proper [interface functions](https://docs.julialang.org/en/v1/manual/interfaces/). Any return type used with this package need to implement the AltInplaceOpsInterface.


## Install

```julia
import Pkg
Pkg.add("DistributedMonteCarlo")
```

## Usage

```julia
using DistributedMonteCarlo
using Distributed

dim = Val(3)
snapshot_type = Float64
return_type = Matrix{Float64}
n_snapshots = 100
tol = 0.01
rndF() = map(x->randn(), 1:3)
mc = MonteCarlo(dim, snapshot_type, return_type, n_snapshots, tol, rndF)
addprocs(1) # you need to add at least one worker
worker_ids = workers()
@everywhere begin
	using StaticArrays # StaticArrays has to be imported since x is a SVector
	sample_func(x, ID::String) = x*x'
end
exp_val = distributed_𝔼(mc, sample_func, worker_ids)
var_val = distributed_var(mc, sample_func, exp_val, worker_ids)
```

