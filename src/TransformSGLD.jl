import Random: seed!
import SpecialFunctions: beta, gamma, erf
import Statistics: mean
import AverageShiftedHistograms: xy, ash
import Hyperopt: @hyperopt, GPSampler, Min
import Plots

# General

function sample(dist, sampler, trans, data)
    param = deepcopy(dist.init_param)
    samples::Array{typeof(param),1} = []
    for d in data
        update!(param, sampler, dist.grad(param,d), trans)
        all(isfinite.(param)) ? push!(samples, deepcopy(param)) : error("inf")
    end
    return samples
end

# Array{Array{Float64,1},1} → Array{Float64,1}
# https://github.com/SciML/ODE.jl/issues/80#issuecomment-154050984
vcat_nosplat(y::Array{Array{Float64,1},1}) =  eltype(y[1])[el[1] for el in y]

# Array{Array{Float64,1},1} → normalized distribution
density(samples::Array{Array{Float64,1},1}, domain) = xy(ash(vcat_nosplat(samples); rng=domain))[2]

RMSE(x, y) = sqrt(mean(@. (x-y)^2 ))

# safe area to evaluate
evalpoints(domain) = range(clamp.(domain,-10.0,10.0)..., length=100)

# The latest PyCall.jl v1.91.4 is not compatible with Optuna v1.5.
# We use the Gaussian Process instead of the tree-structured Parzen estimator.
function optimize_lr_for_pdf(n_trial, lr_range, dist, sampler_name::DataType, trans, data; rseed=1234, showplot=false)
    # evaluate finite interval
    # exclude edges to avoid Inf (e.g. beta)
    xs = evalpoints(dist.domain)[2:end-1]
    # seed must be out of @hyperopt
    seed!(rseed)
    ho = @hyperopt for i=n_trial, sampler=GPSampler(Min), lr=lr_range
        # instantiate sampler from struct name
        sampler = sampler_name(lr)
        samples = sample(dist, sampler, trans, data)
        err = RMSE(dist.pdf.(xs), density(samples, xs))
        err
    end
    if showplot; display(Plots.plot(ho)); end
    min_lr = minimum(ho)[1][1]
    return min_lr
end

# Distributions

mutable struct BetaDist
    name::String
    domain::Tuple{Float64,Float64}
    init_param::Array{Float64,1}
    α::Float64
    β::Float64
    pdf::Function
    grad::Function
end

BetaDist(α, β) = BetaDist(
    "Beta($α,$β)", 
    (0.0, 1.0), 
    [0.5],
    α, 
    β, 
    x -> x^(α-1) * (1-x)^(β-1) / beta(α,β), 
    (x,_) -> (@. - (α-1)/x + (β-1)/(1-x) )
)

mutable struct GammaDist
    name::String
    domain::Tuple{Float64,Float64}
    init_param::Array{Float64,1}
    k::Float64
    θ::Float64
    pdf::Function
    grad::Function
end

GammaDist(k, θ) = GammaDist(
    "Gamma($k,$θ)", 
    (0.0, Inf), 
    [1.0],
    k, 
    θ, 
    x -> 1 / (gamma(k)*θ^k) * x^(k-1) * exp(-x/θ), 
    (x,_) -> (@. - (k-1)/x + 1/θ )
)

mutable struct TnormalDist
    name::String
    domain::Tuple{Float64,Float64}
    init_param::Array{Float64,1}
    pdf::Function
    grad::Function
end

tnormal_ϕ(x) = 1/sqrt(2π) * exp(-x^2 / 2)
tnormal_Φ(x) = 1/2 * (1+erf(x/sqrt(2)))

TnormalDist(l, h) = TnormalDist(
    "Tnormal($l,$h)", 
    (l, h), 
    [(l+h)/2],
    x -> tnormal_ϕ(x) / (tnormal_Φ(h) - tnormal_Φ(l)), 
    (x,_) -> x
);

# Transforms

mutable struct SigmoidTransform
    range::Tuple{Float64,Float64}
    f::Function
    f′::Function
    frac::Function # f″(x) / f′(x)
    g::Function
    g′::Function
    g″::Function
end

sig(x) = 1/(1+exp(-x))

SigmoidTransform(l::Float64, h::Float64) = SigmoidTransform(
    (l, h),
    x -> sig(x) * (h-l) + l,
    x -> sig(x) * (1-sig(x)) * (h-l),
    x -> 1-2*sig(x),
    x -> -log((h-l)/(x-l) - 1),
    x -> 1 / (x-l) / (1 - (x-l)/(h-l)),
    x -> (2*(x-l)/(h-l)-1) / (x-l)^2 / (1-(x-l)/(h-l))^2
)

mutable struct SoftplusTransform
    range::Tuple{Float64,Float64}
    f::Function
    f′::Function
    frac::Function # f″(x) / f′(x)
    g::Function
    g′::Function
    g″::Function
end

SoftplusTransform(_::Float64, _::Float64) = SoftplusTransform(
    (0.0, Inf),
    x -> log(1 + exp(x)),
    x -> sig(x),
    x -> 1 - sig(x),
    x -> log(exp(x)-1),
    x -> 1 / (1-exp(-x)),
    x -> - exp(-x) / (1-exp(-x))^2
)

# Samplers

mutable struct mirrorSGLD
    lr::Float64
end

function update!(x::Array, SGLD::mirrorSGLD, grad, trans)
    x .+= - SGLD.lr * grad + sqrt(2.0*SGLD.lr) * randn(size(x))
    x .= mirror(x, trans.range...)
end

function mirror(x::Array, lo, hi, open=true)
    # nextfloat workaround for CuArrays
    if open
        lo += 1.0e-14
        hi -= 1.0e-14
    end
    # mirroring trick (once)
    x += @. max(0, 2 * (lo - x))
    x -= @. max(0, 2 * (x - hi))
    # if still overstep
    x[x .< lo] .= lo .+ (hi-lo) .* rand(eltype(x), size(x[x .< lo]))
    x[hi .< x] .= lo .+ (hi-lo) .* rand(eltype(x), size(x[hi .< x]))
    return x
end

mutable struct ItoSGLD
    lr::Float64
end

function update!(x::Array, SGLD::ItoSGLD, grad, trans)
    y = @. trans.g(x)
    @. y += SGLD.lr * (-trans.g′(x) * grad + trans.g″(x)) + trans.g′(x) * sqrt(2.0*SGLD.lr) * randn()
    clamp!(y, -10, 10) # avoid Inf
    @. x = trans.f(y)
end

mutable struct CoRVSGLD
    lr::Float64
end

function update!(x::Array, SGLD::CoRVSGLD, grad, trans)
    y = @. trans.g(x)
    @. y += - SGLD.lr * (trans.f′(y) * grad - trans.frac(y)) + sqrt(2.0*SGLD.lr) * randn()
    @. x = trans.f(y)
end