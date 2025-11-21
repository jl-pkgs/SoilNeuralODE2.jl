function get_mock_data()
  # 模拟数据：h (状态), u (驱动)
  # u: [Driver, Time, Batch], h: [Layer, Time, Batch]
  u = rand(Float32, D_INPUT, T_STEPS, BATCH_SZ) .* 0.1f0
  h = zeros(Float32, L_LAYERS, T_STEPS, BATCH_SZ)

  # 初始化 t=1
  h[:, 1, :] .= rand(Float32, L_LAYERS, BATCH_SZ) .* 0.4f0

  # 生成伪真值 (用于训练的 Ground Truth)
  for t in 2:T_STEPS
    rain = u[1:1, t, :]
    # 简化的物理逻辑：上一时刻衰减 + 降雨补给
    h_prev = h[:, t-1, :]
    flow_in = vcat(rain, zeros(Float32, L_LAYERS - 1, BATCH_SZ))
    h[:, t, :] = clamp.(h_prev .* 0.95f0 .+ flow_in, 0f0, 1f0)
  end
  return u, h
end
