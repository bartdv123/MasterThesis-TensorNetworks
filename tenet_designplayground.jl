using Makie
using GLMakie
using Tenet
using LinearAlgebra
using EinExprs
using UUIDs: uuid4

Makie.inline!(true)

make_plot = true

# Create the tensor network, this will allow me to test my methods
seed1 = 6
TN = rand(TensorNetwork, 3::Integer, 4::Integer; out = 0, dim = 2:4, seed = seed1, globalind = false)
println("Tensor network generated=", TN)




indices = inds(TN, :inner)
# index_size_map = size(TN)


if make_plot == true
    println("generating plot")
    # Make a drawing of the tensor network
    # -> Takes a long long time for larger networks...
    node_colors = ["red", "green", "blue", "yellow", "orange"]
    drawing = plot(TN, labels=true)
    display(drawing)
end



function generate_groupedbonds(tn)

    "Generates a dictionary containing as keys the lists of bond-indices and as vals the tensors in connecting the bonds"

    result_dict = Dict{Vector, Vector}()

    for id in inds(tn)
        tensor1, tensor2 = Tenet.select(tn, (Symbol(id)))
        common_indices = intersect(inds(tensor1), inds(tensor2))
        result_dict[common_indices] = [tensor1, tensor2]
    end

    return result_dict

end

dict = generate_groupedbonds(TN)
for key in keys(dict)
    println(key)
end

println(keys(dict))


function grouping_bond(tn, indices_to_group)
    " Matricization of a tensor involves rearranging the data of a tensor into a matrix form based on a specific grouping of indices. "
    " This function takes in a TensorNetwork and a list of indices insize this network which should be grouped together"

    # Compute the dimension of the new bond -> this new bond has the size of the product of the grouped dimensions

    tensor1, tensor2 = Tenet.select(tn, indices_to_group) # take out the tensors which are relevant to this bond
    dim_new = prod(size(tn, index) for index in indices_to_group) # compute the dimension of the new grouped bond

    # compute the indices to leave ungrouped
    complementind_T1 = setdiff(inds(tensor1), indices_to_group)
    complementind_T2 = setdiff(inds(tensor2), indices_to_group)

    # generate a permutation order of indices which puts the to group indices in the end
    # by reshaping the underlying data array -> (dim(complementind_T1)..., dim_new) the correct parts of the data array are grouped

    new_order1 = [complementind_T1..., indices_to_group...] # indices to keep at front
    new_order2 = [complementind_T2..., indices_to_group...] # indices to group at the back
    
    # Permute the dimensions based on the new order
    permuted_tensor1 = permutedims(tensor1, new_order1)
    permuted_tensor2 = permutedims(tensor2, new_order2)

    # extract the permuted tensor data to be reshaped
    new_tensor1_data_array = permuted_tensor1.data
    new_tensor2_data_array = permuted_tensor2.data

    # perform the reshaping and create new tensors with grouped index
    new_index = join([string(index) for index in indices_to_group], "")
    
    # create reshape lists -> casted to tuple for the reshape function
    # reshape lists are based on sizes of the underlying data arrays
    # vcat vertically concatenates two arrays

    tensor1_reshape_list = vcat([size(tensor1, Symbol(id)) for id in complementind_T1], dim_new)
    tensor2_reshape_list = vcat([size(tensor2, Symbol(id)) for id in complementind_T2], dim_new)

    # Create the new tensors based on the the reshaped data array and on a tuple containing the untouched indices and the newly grouped index

    tensor1_new = Tensor(reshape(new_tensor1_data_array, (tensor1_reshape_list...)), (complementind_T1..., Symbol(new_index)))
    tensor2_new = Tensor(reshape(new_tensor2_data_array, (tensor2_reshape_list...)), (complementind_T2..., Symbol(new_index)))

    # Modify the tesnor network by replacing the old tensors with the new tensors where they have a grouped bond
    pop!(tn, tensor1)
    push!(tn, tensor1_new)
    pop!(tn, tensor2)
    push!(tn, tensor2_new)

    indices_mapping = Dict{String, String}()  # Specify the type of the keys and values

    for index in indices_to_group
        indices_mapping[string(index)] = new_index
    end

    println(indices_mapping)
   
end

println(contract(TN))

for indices_to_group in keys(dict)
    if length(indices_to_group) == 1
        continue
    else
        groupbonds(TN, indices_to_group)
    end
end

if make_plot == true
    println("generating plot")
    # Make a drawing of the tensor network
    # -> Takes a long long time for larger networks...
    node_colors = ["red", "green", "blue", "yellow", "orange"]
    drawing = plot(TN, labels=true)
    display(drawing)
end

println(contract(TN))





