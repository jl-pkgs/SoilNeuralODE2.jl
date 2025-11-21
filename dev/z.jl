  # @show ntime
using Lux, Random, Optimisers, Zygote, ComponentArrays, Statistics, Printf

# ==========================================
# 1. 定义各个子组件
# ==========================================

# --- 上边界网络: 下渗 ---
# 输入: [Rain, h_surface]
# 输出: Infiltration Flux (scalar)
# 物理约束: 输出必须 >= 0，且通常 <= Rain (我们这里先保证 >=0)
function build_top_net()
  return Chain(
    Dense(2 => 32, tanh),
    Dense(32 => 1, softplus) # Flux >= 0
  )
end

# --- 下边界网络: 地下水交互 ---
# 输入: [GWL, h_bottom]
# 输出: Bottom Flux (scalar)
# 物理约束: 可以是正(毛细上升)或负(渗漏)，使用 Linear (无激活) 或 tanh
function build_bot_net()
  return Chain(
    Dense(2 => 32, tanh),
    Dense(32 => 1) # Linear output (+/-)
  )
end

# --- 内部再分布网络 ---
# 仅负责土体内部的水分重整，不产生水也不消耗水(理想情况下)
# 输入: h (L)
# 输出: Δh (L)
function build_internal_net(n_layers)
  return Chain(
    Dense(n_layers => 64, tanh),
    Dense(64 => 64, tanh),
    Dense(64 => n_layers) # Linear
  )
end

# --- 汇项网络 (ET & Lateral) ---
# 之前已经定义过，这里复用构建逻辑
function build_sink_net(n_in, n_out)
  return Chain(
    Dense(n_in => 32, tanh),
    Dense(32 => n_out, softplus) # Always non-negative
  )
end

# ==========================================
# 2. 复合模型: BoundaryAwareSoilModel
# ==========================================
struct BoundaryAwareSoilModel{T,B,I,E,L} <: Lux.AbstractExplicitLayer
  top_net::T      # 上边界
  bot_net::B      # 下边界
  internal_net::I # 内部流
  et_net::E       # 蒸散发
  lat_net::L      # 侧向流
  n_layers::Int   # 层数记录
end

# 构造函数
function BoundaryAwareSoilModel(n_layers::Int)
  return BoundaryAwareSoilModel(
    build_top_net(),              # Top: Rain + h_1
    build_bot_net(),              # Bot: GWL + h_L
    build_internal_net(n_layers), # Internal: h_all
    build_sink_net(n_layers + 1, n_layers), # ET: h + ET0
    build_sink_net(n_layers, n_layers),     # Lat: h
    n_layers
  )
end

# 初始化参数
Lux.initialparameters(rng::AbstractRNG, m::BoundaryAwareSoilModel) = (
  top=Lux.initialparameters(rng, m.top_net),
  bot=Lux.initialparameters(rng, m.bot_net),
  int=Lux.initialparameters(rng, m.internal_net),
  et=Lux.initialparameters(rng, m.et_net),
  lat=Lux.initialparameters(rng, m.lat_net)
)
Lux.initialstates(rng::AbstractRNG, m::BoundaryAwareSoilModel) = (
  top=Lux.initialstates(rng, m.top_net),
  bot=Lux.initialstates(rng, m.bot_net),
  int=Lux.initialstates(rng, m.internal_net),
  et=Lux.initialstates(rng, m.et_net),
  lat=Lux.initialstates(rng, m.lat_net)
)

# --- 核心前向传播逻辑 ---
function ((m::BoundaryAwareSoilModel))(inputs, ps, st)
  # inputs: (h_prev, u_curr)
  # u_curr 结构假设: [Rain(1), ET0(1), GWL(1)] -> 3个驱动变量
  h_prev, u_curr = inputs

  Rain = u_curr[1:1, :]
  ET0 = u_curr[2:2, :]
  GWL = u_curr[3:3, :]

  # 1. 计算边界通量
  # Top: 依赖 Rain 和 表层土 h[1]
  h_surf = h_prev[1:1, :]
  x_top = vcat(Rain, h_surf)
  (q_infil_potential, st_top) = m.top_net(x_top, ps.top, st.top)

  # *物理强约束*: 实际入渗量 = min(Potential, Rain)
  # 这保证了模型不会凭空产生比降雨还多的水
  q_infil = min.(q_infil_potential, Rain)

  # Bot: 依赖 GWL 和 底层土 h[end]
  h_bot = h_prev[end:end, :]
  x_bot = vcat(GWL, h_bot)
  (q_bot, st_bot) = m.bot_net(x_bot, ps.bot, st.bot) # +为补给, -为渗漏

  # 2. 计算内部再分布 (Internal Redistribution)
  (dh_int, st_int) = m.internal_net(h_prev, ps.int, st.int)

  # 3. 计算汇项 (ET & Lateral)
  x_et = vcat(h_prev, ET0)
  (q_et, st_et) = m.et_net(x_et, ps.et, st.et)
  (q_lat, st_lat) = m.lat_net(h_prev, ps.lat, st.lat)

  # 4. 组装总变化量 (Mass Balance Assembly)
  # 先初始化总变化为内部变化
  total_delta = dh_int .- q_et .- q_lat

  # 上边界通量加到第一层
  # 注意：Lux 中不建议直接修改数组，这里使用 view 或重建向量会比较安全，
  # 但为了代码极简，且 Zygote 支持这种简单的 array indexing gradient，我们构建 delta 修正项

  # 构建边界通量向量 (Boundary Flux Vector)
  # Top 只有第一层有，Bot 只有最后一层有
  # 创建一个全是0的mask，填充首尾
  zeros_L = zero(total_delta) # 保持维度和类型
  # 这种写法 Zygote 友好度一般，更好的方式是使用 ChainRules 或 简单的拼接
  # 极简方案：使用 vcat 拼接首层、中间层、尾层变化

  # 分解 total_delta
  d_1 = total_delta[1:1, :]
  d_mid = total_delta[2:end-1, :]
  d_end = total_delta[end:end, :]

  # 修正首尾
  d_1_new = d_1 .+ q_infil
  d_end_new = d_end .+ q_bot

  final_delta = vcat(d_1_new, d_mid, d_end_new)

  return final_delta, (top=st_top, bot=st_bot, int=st_int, et=st_et, lat=st_lat)
end

# ==========================================
# 3. 训练过程 (含 GWL 数据)
# ==========================================
function time_step_forward(model, ps, st, u_seq, h_init)
  h_preds = [h_init]
  st_curr = st

  for t in 1:(size(u_seq, 2)-1)
    h_prev = h_preds[end]
    u_curr = view(u_seq, :, t + 1, :) # [Rain, ET, GWL]

    (dh, st_curr) = model((h_prev, u_curr), ps, st_curr)

    push!(h_preds, h_prev .+ dh)
  end
  return stack(h_preds, dims=2), st_curr
end

function get_full_forcing_data(L, T, B)
  # 模拟复杂驱动
  Rain = (rand(Float32, 1, T, B) .> 0.7) .* 0.5f0
  ET0 = rand(Float32, 1, T, B) .* 0.1f0
  GWL = sin.(range(0, 5, length=T))' .* 2f0 # 地下水位波动
  GWL = repeat(Float32.(GWL), 1, 1, B)

  u = vcat(Rain, ET0, GWL) # [3, T, B]
  h = rand(Float32, L, T, B) .* 0.4f0
  return u, h
end

function main()
  L = 6
  T = 100
  B = 16
  rng = Random.default_rng()

  model = BoundaryAwareSoilModel(L)
  ps, st = Lux.setup(rng, model)
  ps_c = ComponentArray(ps)
  opt_state = Optimisers.setup(Optimisers.ADAM(1e-3), ps_c)

  u_train, h_train = get_full_forcing_data(L, T, B)

  println("Training Full Physics-Component Model:")
  println("  1. Top Net (Infiltration <= Rain)")
  println("  2. Bot Net (GWL Interaction)")
  println("  3. Internal Net (Redistribution)")
  println("  4. ET & Lat Nets (Sinks)")
  println("-"^50)

  for epoch in 1:200
    (loss, st), back = Zygote.pullback(ps_c) do p
      h_pred, st_new = time_step_forward(model, p, st, u_train, h_train[:, 1, :])
      mean(abs2, h_pred .- h_train)
    end
    grads = back((one(loss), nothing))[1]
    opt_state, ps_c = Optimisers.update(opt_state, ps_c, grads)

    if epoch % 50 == 0
      @printf "Epoch %3d | Loss: %.6f\n" epoch loss
    end
  end
end

main()
