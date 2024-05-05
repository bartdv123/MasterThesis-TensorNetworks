using Plots
using LinearAlgebra
using Tenet

a = rand(5)
println(typeof(a))
println(typeof(rand(3,3)))
println(typeof(rand(3,3,3)))
a = rand(3,3, 3, 3)

at = Tenet.Tensor(a, (:i, :j, :k, :l))
println(typeof(at))
U, S, V = LinearAlgebra.svd(at, left_inds=[:i])
display(S)
b = rand(3)
bt = Tenet.Tensor(b, [:i])
