using Lux, Enzyme, ComponentArrays, Optimisers, Random
using Statistics

rng = Xoshiro(123)
X = reshape(Float32.(0:0.05:2pi), 1, :) |> collect
Y = sin.(X) .+ 0.02f0 * randn(rng, 1, length(X))
Y = Float32.(Y)

function predict(model, ps, st, X)
  y_pred, st_new = model(X, ps, st)
  return y_pred
end

function loss_func(model, ps, st, X, Y)
  y_pred = predict(model, ps, st, X)
  return mean((y_pred .- Y) .^ 2)
end


model = Chain(Dense(1 => 16, relu), Dense(16 => 1))

function train(model, X, Y; lr=0.005)
  rng = Xoshiro(123)
  ps, st = Lux.setup(rng, model)
  ps = ComponentArray(ps)
  dps = make_zero(ps)

  # 初始化优化器
  opt = Optimisers.Adam(lr)
  opt_state = Optimisers.setup(opt, ps)
  println("Initial Loss: ", loss_func(model, ps, st, X, Y,))

  for epoch in 1:2000
    dps .= 0 # 梯度清零
    Enzyme.autodiff(Reverse, loss_func,
      Const(model), Duplicated(ps, dps), Const(st), Const(X), Const(Y)) # 计算梯度
    opt_state, ps = Optimisers.update(opt_state, ps, dps) # 更新参数

    if epoch % 200 == 0
      loss = loss_func(model, ps, st, X, Y)
      println("Epoch $epoch: Loss = $loss")
    end
  end
  println("Final Loss: ", loss_func(model, ps, st, X, Y))
  ypred = predict(model, ps, st, X)
  return ypred
end


begin
  ypred = train(model, X, Y; lr=0.01)

  using Plots
  plot(X[:], Y[:], label="True")
  plot!(X[:], ypred[:], label="Predicted")
  savefig("lux_model_fit.png")
end
