using Lux, Random, Optimisers, ComponentArrays, Statistics, Printf
using Enzyme

# ==========================================
# 1. 配置与数据生成 (Configuration & Data)
# ==========================================
const L_LAYERS = 10      # 土壤层数
const T_STEPS = 50       # 时间步长
const BATCH_SZ = 32      # 批次大小
const D_INPUT = 2        # 驱动变量: [降雨, 蒸发]

include("build_forcing_dummy.jl")
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

# ==========================================
# 3. 前向传播与时间步进 (Forward Pass)
# ==========================================
# 显式展开时间循环 (Unroll)，模拟物理递推
function time_step_forward(model, ps, st, u_seq, h_init)
  # u_seq: [In, T, B], h_init: [L, B]
  ntime = size(u_seq, 2)
  h_preds = Vector{Matrix{eltype(h_init)}}(undef, ntime)
  h_preds[1] = h_init
  st_curr = st

  # 从 t=1 推导 t=2, ..., T
  for t in 1:ntime-1
    h_prev = h_preds[t]
    u_curr = view(u_seq, :, t + 1, :)

    x_in = vcat(h_prev, u_curr) # 拼接: [土壤状态; 气象驱动]
    dh, st_curr = model(x_in, ps, st_curr)
    h_preds[t+1] = h_prev .+ dh
  end

  # 堆叠为张量: [L, T, B]
  return stack(h_preds, dims=2), st_curr
end

# ==========================================
# 4. 训练核心 (Training Core)
# ==========================================
function loss_function(model, ps, st, u_data, h_true)
  # 使用数据的 t=1 时刻作为初始条件
  h_init = view(h_true, :, 1, :)
  (h_pred, st_new) = time_step_forward(model, ps, st, u_data, h_init) # 预测整个序列
  
  loss = mean(abs2, h_pred .- h_true) # 计算 MSE Loss
  return loss, st_new, ()
end


function main()
  rng = Random.default_rng()

  # 初始化
  net = build_network()
  ps, st = Lux.setup(rng, net)
  ps_c = ComponentArray(ps) # 参数扁平化
  dps = zero(ps_c) # 梯度缓存

  opt = Optimisers.ADAM(1e-3)
  opt_state = Optimisers.setup(opt, ps_c)

  # 数据
  println("Start Training: Layers=$(L_LAYERS), Steps=$(T_STEPS)")
  println("-"^30)

  # 包装函数，仅返回 loss 标量
  function compute_loss(p, model, state, u, h)
    l, _, _ = loss_function(model, p, state, u, h)
    return l
  end

  for epoch in 1:1000
    # 梯度清零
    dps .= 0
    
    # Enzyme 自动微分
    # 注意: Enzyme 需要明确传入 Const/Duplicated
    Enzyme.autodiff(Reverse, compute_loss, Active, 
                    Duplicated(ps_c, dps), 
                    Const(net), 
                    Const(st), 
                    Const(forcing), 
                    Const(θ_obs))
    
    # 参数更新
    opt_state, ps_c = Optimisers.update(opt_state, ps_c, dps)

    if epoch % 50 == 0
      loss = compute_loss(ps_c, net, st, forcing, θ_obs)
      @printf "Epoch %3d | Loss: %.6f\n" epoch loss
    end
  end
end

main()
