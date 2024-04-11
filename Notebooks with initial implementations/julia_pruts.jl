using Plots
using LinearAlgebra

size = 50
x = rand(size,size)
n = [i for i in 1:size]
s = svdvals(x)
s_max = maximum(s)
display(scatter(n, s/s_max, yscale=:log10))
#display(scatter(n[1:5], s[1:5]/s_max, yscale=:log10))