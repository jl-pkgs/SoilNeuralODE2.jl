using Lux, Random, Optimisers, ComponentArrays, Statistics, Printf
using MLUtils  # DataLoader
using Enzyme
using RTableTools
include("Framework.jl")

f = "data/SM_AR_Batesville_8_WNW_2024.csv"
df = fread(f)

# [n_layer, ntime], depth: 5, 10, 20, 50, 100 cm
θ_obs = Matrix(df[:, 3:end])' |> collect .|> Float32
forcing = Matrix(df[:, [:P_CALC]])' |> collect .|> Float32 # [1, ntime]
X, Y = forcing, θ_obs


# 3. 前向传播与时间步进 (Forward Pass)
# 显式展开时间循环 (Unroll)，模拟物理递推
function predict(model, ps, st, forcing, θ_init)
  # u_seq: [In, T, B], h_init: [L, B]
  T = eltype(θ_init)
  ntime = size(forcing, 2)
  L_LAYERS = length(θ_init)
  # h_preds = Array{T}(undef, L_LAYERS, ntime, BATCH_SZ)
  h_preds = Array{T}(undef, L_LAYERS, ntime)
  h_preds[:, 1] .= θ_init
  st_curr = st

  for t in 1:ntime-1
    h_prev = h_preds[:, t]
    u_curr = forcing[:, t+1]

    x_in = vcat(h_prev, u_curr) # 拼接: [土壤状态; 气象驱动]
    dh_vert, st_vert = model.vert(x_in, ps.vert, st_curr.vert) # Vertical 
    dh_lat, st_lat = model.lat(h_prev, ps.lat, st_curr.lat) # Lateral 

    r = h_prev .+ dh_vert .+ dh_lat # 叠加垂直和侧向变化
    h_preds[:, t+1] .= r
    st_curr = (vert=st_vert, lat=st_lat)
  end
  return h_preds, st_curr
end


# 定义垂直流网络拟合变化量: Δh_vert = Net(h_prev, u_curr)
function build_network(; n_layers=5, n_in=1, scale=0.01f0)
  in_dim = n_layers + n_in
  return Chain(
    Dense(in_dim => 64, tanh),
    Dense(64 => 64, tanh),
    Dense(64 => n_layers),  # 输出每层的含水量变化 Δh
    x -> x .* scale       # 缩放输出，防止初始梯度爆炸
  )
end

# 定义侧向流网络: Δh_lat = Net(h_prev)
# 侧向流通常只取决于当前的水分状态（和地形，此处隐含在参数中）
function build_lateral_network(; n_layers=5, scale=0.01f0)
  return Chain(
    Dense(n_layers => 32, tanh),
    Dense(32 => 32, tanh),
    Dense(32 => n_layers, init_weight=zeros32, init_bias=zeros32),
    x -> abs.(x) .* -scale
  )
end


# TODO: 分步冻结，分步训练不同的模块
begin
  scale = 0.001f0
  model = (
    vert=build_network(; scale),
    lat=build_lateral_network(; scale)
  )
  rng = Random.Xoshiro(1) # Use a seeded local RNG for reproducibility
  ps, st = Lux.setup(rng, model) #

  @time ypred = train(X, Y, model; predict, lr=0.0015,
    nepoch=3000, step=50, batchsize=-1)[1] # 不划分batchsize, 需要训练很多次
end


if true
  using Plots
  function plot_layer(i)
    plot(θ_obs[i, :], label="Observed", xlabel="Layer $i", ylabel="Soil Moisture")
    plot!(ypred[i, :], label="Predicted")
  end

  ps = map(plot_layer, 1:5)
  p = plot(ps..., layout=(3, 2), size=(800, 600))
  savefig("Figure03_SM_Enzyme.png")
end
