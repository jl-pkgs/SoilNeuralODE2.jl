using Lux, Random, Optimisers, Zygote, ComponentArrays, Statistics, Printf
# using Enzyme

# ==========================================
# 1. 配置与数据生成 (Configuration & Data)
# ==========================================
const L_LAYERS = 10      # 土壤层数
const T_STEPS = 50       # 时间步长
const BATCH_SZ = 32      # 批次大小
const D_INPUT = 2        # 驱动变量: [降雨, 蒸发]


# ==========================================
# 2. 模型定义 (Model Architecture)
# ==========================================
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
  # 预分配存储空间不仅是为了性能，也是为了收集计算图
  # Use Zygote.Buffer to allow mutation for AD, and AbstractMatrix to handle both SubArray and Array
  h_preds = Zygote.Buffer(Vector{AbstractMatrix{eltype(h_init)}}(undef, size(u_seq, 2)))
  h_preds[1] = h_init

  st_curr = st
  ntime = size(u_seq, 2)
  
  # 从 t=1 推导 t=2, ..., T
  for t in 1:ntime-1
    h_prev = h_preds[t]
    u_curr = view(u_seq, :, t + 1, :)

    # 输入拼接: [土壤状态; 气象驱动]
    x_in = vcat(h_prev, u_curr)

    # 计算增量 Δh
    (dh, st_curr) = model(x_in, ps, st_curr)

    # 状态更新: h_t = h_{t-1} + Δh
    # @show size(dh)
    # @show size(h_prev)
    # @show size(h_preds), size(h_preds[t])

    h_preds[t+1] = h_prev .+ dh
  end

  # 堆叠为张量: [L, T, B]
  return stack(copy(h_preds), dims=2), st_curr
end

# ==========================================
# 4. 训练核心 (Training Core)
# ==========================================
function loss_function(model, ps, st, u_data, h_true)
  # 使用数据的 t=1 时刻作为初始条件
  h_init = view(h_true, :, 1, :)

  # 预测整个序列
  (h_pred, st_new) = time_step_forward(model, ps, st, u_data, h_init)

  # 计算 MSE Loss
  loss = mean(abs2, h_pred .- h_true)
  return loss, st_new, ()
end


function main()
  rng = Random.default_rng()

  # 初始化
  net = build_network()
  ps, st = Lux.setup(rng, net)
  ps_c = ComponentArray(ps) # 参数扁平化

  opt = Optimisers.ADAM(1e-3)
  opt_state = Optimisers.setup(opt, ps_c)

  # 数据
  println("Start Training: Layers=$(L_LAYERS), Steps=$(T_STEPS)")
  println("-"^30)

  p = ps_c
  loss_func(p) = loss_function(net, p, st, forcing, θ_obs)
  
  # loss_function(net, p, st, u_train, θ_obs)
  for epoch in 1:1000
    # 自动微分 (Reverse Mode)
    (loss, st), back = Zygote.pullback(loss_func, ps_c)
    grads = back((one(loss), nothing, nothing))[1]

    opt_state, ps_c = Optimisers.update(opt_state, ps_c, grads)

    if epoch % 50 == 0
      @printf "Epoch %3d | Loss: %.6f\n" epoch loss
    end
  end
end

# 执行
main()
