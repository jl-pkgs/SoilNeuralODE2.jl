# 这里带入了真实观测值
# kongdd, 20251121
using Lux, Random, Optimisers, ComponentArrays, Statistics, Printf
using Enzyme
using RTableTools

function of_NSE(obs, sim)
  top = sum((sim .- obs) .^ 2)
  bot = sum((obs .- mean(obs)) .^ 2)
  return 1 - (top / bot)
end

## 带入真实观测
f = "data/SM_AR_Batesville_8_WNW_2024.csv"
df = fread(f)

forcing = Matrix(df[:, [:P_CALC]])' |> collect .|> Float32 # [1, ntime]
θ_obs = Matrix(df[:, 3:end])' |> collect .|> Float32 # [n_layer, ntime], depth: 5, 10, 20, 50, 100 cm

# 1. 配置与数据生成 (Configuration & Data)
const L_LAYERS = 5      # 土壤层数
# const BATCH_SZ = 32      # 批次大小
# const D_INPUT = 2        # 驱动变量: [降雨, 蒸发]


# 定义网络拟合变化量: Δh = Net(h_prev, u_curr)
function build_network(; n_layers=5, n_in=1)
  in_dim = n_layers + n_in
  return Chain(
    Dense(in_dim => 64, tanh),
    Dense(64 => 64, tanh),
    Dense(64 => n_layers),  # 输出每层的含水量变化 Δh
    x -> x .* 0.01f0        # 缩放输出，防止初始梯度爆炸
  )
end

# 3. 前向传播与时间步进 (Forward Pass)
# 显式展开时间循环 (Unroll)，模拟物理递推
function time_step_forward(model, ps, st, forcing, θ_init)
  # u_seq: [In, T, B], h_init: [L, B]
  T = eltype(θ_init)
  ntime = size(forcing, 2)
  # h_preds = Array{T}(undef, L_LAYERS, ntime, BATCH_SZ)
  h_preds = Array{T}(undef, L_LAYERS, ntime)
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


# predict(model, ps, st, forcing, θ_obs[:, 1])
function predict(model, ps, st, forcing, θ_init)
  h_pred, _ = time_step_forward(model, ps, st, forcing, θ_init)
  return h_pred
end

# 4. 训练核心 (Training Core)
function loss_function(model, ps, st, forcing, θ_obs)
  # 使用数据的 t=1 时刻作为初始条件
  h_init = @view θ_obs[:, 1]
  (h_pred, st_new) = time_step_forward(model, ps, st, forcing, h_init) # 预测整个序列
  return -of_NSE(θ_obs, h_pred)
  # loss = mean(abs2, h_pred .- θ_obs) # 计算 MSE Loss
  # return loss
end


function main(; lr=0.0005, nepoch=1000, step=50)
  rng = Random.Xoshiro(1) # Use a seeded local RNG for reproducibility

  # 初始化
  model = build_network()
  ps, st = Lux.setup(rng, model)
  ps_c = ComponentArray(ps) # 参数扁平化
  dps = zero(ps_c) # 梯度缓存

  opt = Optimisers.ADAM(lr)
  opt_state = Optimisers.setup(opt, ps_c)

  # 数据
  println("Start Training: Layers=$(L_LAYERS), Steps=$(size(forcing, 2))")
  println("-"^30)

  # 包装函数，仅返回 loss 标量
  # compute_loss(p, model, state, u, h) = loss_function(model, p, state, u, h)
  compute_loss(p) = loss_function(model, p, st, forcing, θ_obs)

  for epoch in 1:nepoch
    dps .= 0
    # 使用 set_runtime_activity(Reverse) 解决 "Constant memory is stored ... to a differentiable variable" 错误
    Enzyme.autodiff(
      Enzyme.set_runtime_activity(Reverse), 
      # Reverse, 
      compute_loss, Active,
      Duplicated(ps_c, dps),
    )

    opt_state, ps_c = Optimisers.update(opt_state, ps_c, dps) # 参数更新
    if epoch % step == 0
      loss = compute_loss(ps_c)
      @printf "Epoch %3d | Loss: %.6f\n" epoch loss
    end
  end

  ypred = predict(model, ps_c, st, forcing, θ_obs[:, 1])
  return ypred
end

@time ypred = main(; lr=0.005, nepoch=5000, step=100)
# 1.893517 seconds (4.55 M allocations: 4.220 GiB, 20.33% gc time)

begin
  using Plots
  function plot_layer(i) 
    plot(θ_obs[i, :], label="Observed", xlabel="Layer $i", ylabel="Soil Moisture")
    plot!(ypred[i, :], label="Predicted")
  end

  ps = map(plot_layer, 1:5)
  p = plot(ps..., layout=(3,2), size=(800,600))
  savefig("Figure02_SM_Enzyme.png")
end
