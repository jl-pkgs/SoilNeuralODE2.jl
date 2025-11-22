using Random, Lux, Optimisers, ComponentArrays, Statistics, Printf
using MLUtils  # DataLoader
using Enzyme


of_MSE(obs, sim) = mean(abs2, sim .- obs)

function of_NSE(obs, sim)
  top = sum((sim .- obs) .^ 2)
  bot = sum((obs .- mean(obs)) .^ 2)
  return 1 - (top / bot)
end

function loss_function(model, ps, st, X, Y; predict::Function)
  h_pred, = predict(model, ps, st, X, Y[:, 1])
  return -of_NSE(Y, h_pred) # of_MSE(Y, h_pred)
end


function train(X, Y, model; predict::Function, 
  lr=0.0005, nepoch=1000, step=50, batchsize=-1)

  batchsize > 0 && (loader_train = DataLoader((; X, Y); batchsize, shuffle=true))

  rng = Random.Xoshiro(1) # Use a seeded local RNG for reproducibility
  ps, st = Lux.setup(rng, model)
  ps_c = ComponentArray(ps) # 参数扁平化
  dps = Enzyme.make_zero(ps_c) # 梯度缓存

  opt = Optimisers.ADAM(lr)
  opt_state = Optimisers.setup(opt, ps_c)

  L_LAYERS = size(Y, 1)
  ntime = size(X, 2)
  println("Start Training: Layers=$(L_LAYERS), Steps=$(ntime)")
  println("-"^30)
  compute_loss(p, X, Y) = loss_function(model, p, st, X, Y; predict)

  function grads!(dps, xb, yb)
    dps .= 0
    Enzyme.autodiff(
      Enzyme.set_runtime_activity(Reverse),
      compute_loss, Active,
      Duplicated(ps_c, dps), Const(xb), Const(yb)
    )
  end

  for epoch in 1:nepoch
    if batchsize > 0
      for (xb, yb) in loader_train
        grads!(dps, xb, yb) # 计算梯度
        opt_state, ps_c = Optimisers.update(opt_state, ps_c, dps) # 参数更新
      end
    else
      grads!(dps, X, Y) # 计算梯度
      opt_state, ps_c = Optimisers.update(opt_state, ps_c, dps) # 参数更新
    end

    if epoch % step == 0
      loss = compute_loss(ps_c, forcing, θ_obs)
      @printf "Epoch %3d | Loss: %.6f\n" epoch loss
    end
  end

  ypred, st_new = predict(model, ps_c, st, forcing, θ_obs[:, 1])
  return ypred, model, ps_c
end
