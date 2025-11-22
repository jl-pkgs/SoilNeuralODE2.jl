# 带入了真实观测值
# kongdd, 20251121
using Lux, Random, Optimisers, ComponentArrays, Statistics, Printf
using MLUtils  # DataLoader
using Enzyme
using RTableTools
include("Framework.jl")
include("LSTM.jl")

f = "data/SM_AR_Batesville_8_WNW_2024.csv"
df = fread(f)

# [n_layer, ntime], depth: 5, 10, 20, 50, 100 cm
θ_obs = Matrix(df[:, 3:end])' |> collect .|> Float32
forcing = Matrix(df[:, [:P_CALC]])' |> collect .|> Float32 # [1, ntime]
X, Y = forcing, θ_obs


# 显式展开时间循环 (Unroll)，模拟物理递推
function predict(model, ps, st, forcing, θ_init)
  # u_seq: [In, T, B], h_init: [L, B]
  T = eltype(θ_init)
  ntime = size(forcing, 2)
  nlayer = length(θ_init)
  # h_preds = Array{T}(undef, L_LAYERS, ntime, BATCH_SZ)
  h_preds = Array{T}(undef, nlayer, ntime)
  h_preds[:, 1] .= θ_init
  st_curr = st

  for t in 1:ntime-1
    h_prev = h_preds[:, t]
    u_curr = forcing[:, t+1]

    x_in = vcat(h_prev, u_curr) # 拼接: [土壤状态; 气象驱动]
    dh, st_curr = model(x_in, ps, st_curr)
    r = h_prev .+ dh
    h_preds[:, t+1] .= r
  end
  return h_preds, st_curr
end

# 定义网络拟合变化量: Δh = Net(h_prev, u_curr)
model = Chain(
  Dense(6 => 64, tanh),
  Dense(64 => 64, tanh),
  Dense(64 => 5),   # 输出每层的含水量变化 Δh
  x -> x .* 001f0   # 缩放输出，防止初始梯度爆炸
)

model = Chain(
  # Recurrence(LSTMCell(6 => 64), return_sequence=true), # (n_layer=5) + (n_in=1) = 5
  LSTM(6 => 64), # n_layer=5
  LSTM(64 => 64),
  Dense(64 => 5),
  x -> x .* 0.001f0
)

begin
  x = rand(Float32, 6, 128)
  ps, st = Lux.setup(rng, model)
  y, st_new = model(x, ps, st)
end


# model = build_network(; scale=0.001f0)
rng = Random.Xoshiro(1) # Use a seeded local RNG for reproducibility
ps, st = Lux.setup(rng, model)

@time ypred = train(X, Y, model; predict, lr=0.002,
  nepoch=2000, step=50, batchsize=-1)[1] # 不划分batchsize, 需要训练很多次
# 1.893517 seconds (4.55 M allocations: 4.220 GiB, 20.33% gc time)

# @time ypred = train(data...; lr=0.002, nepoch=1000, step=100,
#   scale=0.001f0, batchsize=24*30)[1] # 采用batchsize训练不成功

if true
  using Plots
  gr(framestyle=:box)

  function plot_layer(i)
    plot(θ_obs[i, :], label="Observed", xlabel="Layer $i", ylabel="Soil Moisture")
    plot!(ypred[i, :], label="Predicted")
  end

  ps = map(plot_layer, 1:5)
  p = plot(ps..., layout=(3, 2), size=(800, 600))
  savefig("Figure02_SM_LSTM.png")
end
