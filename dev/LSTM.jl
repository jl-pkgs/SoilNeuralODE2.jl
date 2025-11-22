using Lux, Random
using ConcreteStructs # 既然你之前问过，这里正好用上
import Lux

# 定义结构体：它是一个 Lux 层，内部包含一个 cell
@concrete struct LSTM <: AbstractLuxLayer
  cell
end

# 构造函数：方便你像 LSTMCell 一样直接传参数 (例如 6 => 64)
function LSTM(dims::Pair{Int,Int}; kwargs...)
  return LSTM(LSTMCell(dims; kwargs...))
end

# 前向传播 (Forward Pass)
# 这是核心逻辑：合并了计算和提取
function (l::LSTM)(x, ps, st)
  # 1. 调用内部的 LSTMCell
  # 注意：因为 cell 是结构体的一个字段，所以它的参数在 ps.cell，状态在 st.cell
  (y_tuple, new_st_inner) = l.cell(x, ps.cell, st.cell)

  # 2. 提取 h (y_tuple[1]), 扔掉 (h, c)
  h = y_tuple[1]

  # 3. 返回 h 和更新后的状态 (Lux 会自动保持 st 的结构)
  return h, (cell=new_st_inner,)
end

function Lux.initialparameters(rng::AbstractRNG, l::LSTM)
  return (cell=Lux.initialparameters(rng, l.cell),)
end

function Lux.initialstates(rng::AbstractRNG, l::LSTM)
  return (cell=Lux.initialstates(rng, l.cell),)
end

# begin
#   model = Chain(LSTM(6 => 64), LSTM(64 => 32))
#   x = rand(Float32, 6, 128)
#   ps, st = Lux.setup(rng, model)
#   y, st_new = model(x, ps, st)
# end
