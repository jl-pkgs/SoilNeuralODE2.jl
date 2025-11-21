using Lux, Random, Optimisers, ComponentArrays, Statistics, Printf
using Enzyme

# 1. 配置与数据生成 (Configuration & Data)
const L_LAYERS = 10      # 土壤层数
const T_STEPS = 50       # 时间步长
const BATCH_SZ = 32      # 批次大小
const D_INPUT = 2        # 驱动变量: [降雨, 蒸发]

include("build_forcing_dummy.jl")
Random.seed!(123) # Ensure data generation is deterministic
forcing, θ_obs = get_mock_data()


# 定义网络拟合变化量: Δh = Net(h_prev, u_curr)
function build_network()
  in_dim = L_LAYERS + D_INPUT
  return Chain(
    Dense(in_dim => 64, tanh),
    Dense(64 => 64, tanh),
    Dense(64 => L_LAYERS)  # 输出每层的含水量变化 Δh
  )
end

# 3. 前向传播与时间步进 (Forward Pass)
# 显式展开时间循环 (Unroll)，模拟物理递推
function time_step_forward(model, ps, st, u_seq, h_init)
  # u_seq: [In, T, B], h_init: [L, B]
  T = eltype(h_init)
  ntime = size(u_seq, 2)
  h_preds = Array{T}(undef, L_LAYERS, ntime, BATCH_SZ)
  # h_preds = Vector{Matrix{eltype(h_init)}}(undef, ntime)
  h_preds[:, 1, :] .= h_init
  st_curr = st

  # 从 t=1 推导 t=2, ..., T
  for t in 1:ntime-1
    h_prev = h_preds[:, t, :]
    u_curr = u_seq[:, t+1, :]

    x_in = vcat(h_prev, u_curr) # 拼接: [土壤状态; 气象驱动]
    dh, st_curr = model(x_in, ps, st_curr)
    # h_preds[t+1] = h_prev .+ dh
    r = h_prev .+ dh
    h_preds[:, t+1, :] .= r
  end

  # 堆叠为张量: [L, T, B]
  # return stack(h_preds, dims=2), st_curr
  return h_preds, st_curr
end

# 4. 训练核心 (Training Core)
function loss_function(model, ps, st, u_data, h_true)
  # 使用数据的 t=1 时刻作为初始条件
  h_init = view(h_true, :, 1, :)
  (h_pred, st_new) = time_step_forward(model, ps, st, u_data, h_init) # 预测整个序列

  loss = mean(abs2, h_pred .- h_true) # 计算 MSE Loss
  return loss
end


function main(; lr=0.0005)
  rng = Random.Xoshiro(1) # Use a seeded local RNG for reproducibility

  # 初始化
  model = build_network()
  ps, st = Lux.setup(rng, model)
  ps_c = ComponentArray(ps) # 参数扁平化
  dps = zero(ps_c) # 梯度缓存

  opt = Optimisers.ADAM(lr)
  opt_state = Optimisers.setup(opt, ps_c)

  # 数据
  println("Start Training: Layers=$(L_LAYERS), Steps=$(T_STEPS)")
  println("-"^30)

  # 包装函数，仅返回 loss 标量
  # compute_loss(p, model, state, u, h) = loss_function(model, p, state, u, h)
  compute_loss(p) = loss_function(model, p, st, forcing, θ_obs)

  for epoch in 1:1000
    dps .= 0
    # 使用 set_runtime_activity(Reverse) 解决 "Constant memory is stored ... to a differentiable variable" 错误
    Enzyme.autodiff(
      Enzyme.set_runtime_activity(Reverse), 
      # Reverse, 
      compute_loss, Active,
      Duplicated(ps_c, dps),
      # Const(model),
      # Const(st),
      # Const(forcing),
      # Const(θ_obs)
    )

    opt_state, ps_c = Optimisers.update(opt_state, ps_c, dps) # 参数更新
    if epoch % 50 == 0
      loss = compute_loss(ps_c)
      @printf "Epoch %3d | Loss: %.6f\n" epoch loss
    end
  end
end

@time main(;lr=1e-3)
# 1.893517 seconds (4.55 M allocations: 4.220 GiB, 20.33% gc time)
