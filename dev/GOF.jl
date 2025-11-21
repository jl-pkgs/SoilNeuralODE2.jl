using Statistics

function of_NSE(obs, sim)
  top = sum((sim .- obs) .^ 2)
  bot = sum((obs .- mean(obs)) .^ 2)
  return 1 - (top / bot)
end
