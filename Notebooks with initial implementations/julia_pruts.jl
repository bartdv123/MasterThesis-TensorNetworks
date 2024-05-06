using Plots
using LinearAlgebra
using Tenet
using TensorKit
using MPSKit
using Makie
using GraphMakie.NetworkLayout
using CairoMakie

# b = rand(2,2,2,2,2,2)
# #MPSKit.decompose_local_mps(b)
# bt = TensorMap(b, ℝ^2 ⊗ ℝ ^2 ⊗ ℝ ^2, ℝ^2 ⊗ ℝ ^2 ⊗ ℝ ^2)

# global ψ_small = 0
# for (D, d, elt) in [(ℝ^10, ℝ^2, Float64), (Rep[U₁](-1 => 3, 0 => 3, 1 => 3), Rep[U₁](-1 => 1, 0 => 1, 1 => 1), Float64)]
#     global ψ_small = FiniteMPS(rand, elt, 4, d, D)
# end
# println(ψ_small)
# ψ_large =  MPSKit.decompose_localmps(convert(TensorKit.TensorMap, ψ_small))

b = Tenet.Tensor(rand(Complex{Float64}, 2, 2, 2, 2), (:a, :b, :c, :d))

function decompose_tensor_to_mps(b)
    println("new random tensor \n --------------------------------------------------------------------------------------")
    """
    tensor b represents the contracted loop here,
    i want to represent this contracted loop as an MPS network as faithfully as 
    possible.
    """
    virtual = [Symbol("v$i") for i in 1:100] 
    T_QR1 = []
    R = b
    for (i, index) in enumerate(inds(R))
        if i == length(inds(b))

            push!(T_QR1, R)
            break
        end
        if i == 1
            Q, R = LinearAlgebra.qr(R, left_inds=[index], virtualind = virtual[i])
            push!(T_QR1, Q)
            global prop_id = setdiff(inds(Q), [index])[1]
        end
        if i > 1
            Q, R = LinearAlgebra.qr(R, left_inds=[index, prop_id], virtualind = virtual[i])
            push!(T_QR1, Q)
            global prop_id = setdiff(inds(Q), [index, prop_id])[1]
        end
    end
    return Tenet.TensorNetwork(T_QR1)
end


function decompose_tensor_to_mps_trial(b)
    t_net = decompose_tensor_to_mps(b)
    # drawing1 = Makie.plot(t_net, node_color=[:darkred for i in 1:length(Tenet.tensors(t_net))], labels=true, layout=Stress(), edge_color=:grey80)
    # display(drawing1)
    new_b = Tenet.contract(t_net)
    overlap = abs(Tenet.contract(b, new_b)[1])
    overlap_bb = abs(Tenet.contract(b, b)[1])
    push!(t_net, b)
    overlap_full = (abs((Tenet.contract(t_net))[1]))^2

    
    println("< b | b > = ", overlap_bb)
    println("< b | contracted(QR_b) > = ", overlap)
    println("< b | QR_b > = ", overlap_full/(overlap*overlap_bb))

end
    




decompose_tensor_to_mps_trial(b)