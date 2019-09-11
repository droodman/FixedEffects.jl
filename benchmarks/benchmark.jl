using FixedEffects, CategoricalArrays, Random
Random.seed!(1234)
N = 10000000
K = 100
id1 = categorical(Int.(rand(1:(N/K), N)))
id2 = categorical(Int.(rand(1:K, N)))
x = rand(N)
X = rand(N, 10)
@time solve_residuals!(x, [FixedEffect(id1), FixedEffect(id2)])
#  0.425070 seconds (138 allocations: 235.102 MiB, 2.92% gc time)
@time solve_residuals!(X, [FixedEffect(id1), FixedEffect(id2)])
# 2.873322 seconds (793 allocations: 235.122 MiB, 0.53% gc time)
