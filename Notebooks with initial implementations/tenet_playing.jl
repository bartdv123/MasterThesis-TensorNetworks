#Adding the needed depedencies
using Makie
using CairoMakie
using MakieCore
using Tenet
using LinearAlgebra
using EinExprs

Makie.inline!(true)



contraction_sequence = EinExprs.einexpr(TN, optimizer=EinExprs.Greedy)
ordered_contraction_indices = EinExprs.inds(contraction_sequence)


function size_up_and_reconnect(S, index)
    """ This helper function takes in tensor S from the SVD and splits it in two -> two times s^(1/2) reconnects them with the index of the bond """

    s12_data = S.data.^(1/2)
    s12_data_matrix = LinearAlgebra.diagm(s12_data)
    s12_indices = [inds(S)..., Symbol(index)]
    s12_tensor = Tensor(s12_data_matrix, (s12_indices...,))

    return s12_tensor

end

function tensor_cutting_index!(tensor, index, bondsize)
    """ Function which takes in a tensor, a symbolic index and a bondsize. Reduces the dimension of this index towards bondsize """
    
    slice_location = dim(tensor, index)
    # Create a tuple of slices for each dimension
    slices = [1:size(tensor, dim) for dim in 1:ndims(tensor)]
    # Replace the slice for the specified dimension with the desired range
    slices[slice_location] = 1:bondsize
    # Create a new tensor with the same indices, but the data is rescaled along one dimension
    tensor = Tensor(tensor[slices...], inds(tensor))
    return tensor

end

function generate_groupedbonds(tn)

    """ Generates a dictionary containing as keys lists of bond-indices between two tensors and as vals the tensors in connecting the bonds """

    result_dict = Dict{Vector, Vector}()

    for id in inds(tn)
        tensor1, tensor2 = Tenet.select(tn, (Symbol(id)))
        common_indices = intersect(inds(tensor1), inds(tensor2))
        result_dict[common_indices] = [tensor1, tensor2]
    end

    return keys(result_dict)

end

function truncated_SVD_replacement(tn, index, bondsize, printing=false)

    """ Function which takes in a TensorNetwork and an index which connect two tensors inside of this network.
    This function applies truncated SVD bond-compression through grouping of R-tensors from QR decomposition on the bond.
    Return the TensorNetwork with relevant modification made to bond tensors connected to this index.

    Graphical description of what the code does:

    == ~ Initial bond
    -- ~ Truncated bond

    A  == B
    Q1 == R1 == R2 == Q2 
    Q1 == R12 == Q2 
    Q1 == U == S == Vt == Q2 
    [Q1 == U  -- s12] -- [s12 -- Vt == Q2] 
    new_A -- new_B

    """ 
    
    # select both tensors connected to the index
    tensor1, tensor2 = Tenet.select(tn, (Symbol(index)))
    
    if length(inds(tensor1)) == 1
        println("One of the tensors has no non shared indices. Exiting function.")
        return  # This will exit the function prematurely
    end
    if length(inds(tensor2)) == 1
        println("One of the tensors has no non shared indices. Exiting function.")
        return  # This will exit the function prematurely
    end

    if printing == true
        println(inds(tensor1))
        println(inds(tensor2))
        println("bond index=", intersect(inds(tensor1), inds(tensor2)))
        println("The size of tensor1 =", size(tensor1))
        println("The size of tensor2 =", size(tensor2))
    end

    # extract the common and complimentairy indices
    common_index = Set([(index)])  # Convert to Set
    complementind_T1 = setdiff(inds(tensor1), common_index)
    complementind_T2 = setdiff(inds(tensor2), common_index)



    if printing == true
        println("complement ind T1 and T2")
        println(complementind_T1)
        println(complementind_T2)
    end


    # Perform a QR-decomposition on both tensors in the bond in a way that the R -factors are pushed towards the middle of the bond
    Q1, R1 = LinearAlgebra.qr(tensor1, left_inds = (complementind_T1...,))
    Q2, R2 = LinearAlgebra.qr(tensor2, left_inds = (complementind_T2...,))

    if printing == true
        # sizes and corresponding indices in the Q, R tensors
        println("Q1 Tensor=", size(Q1), inds(Q1), "R1 Tensor=", size(R1), inds(R1))
        println("Q2 Tensor=", size(Q2), inds(Q2), "R2 Tensor=", size(R2), inds(R2))
    end

    # Perform a SVD-decomposition on the R12 tensor -> truncation of the S - matrix will allow one to make an approximation to the network

    new_bond = string(index)*"_*"




    R12 = contract(R1, R2)
    Q1_inds = intersect(inds(Q1), inds(R12)) #extract the indices pointing towards Q1
    U, S, Vt = LinearAlgebra.svd(R12, left_inds = (Q1_inds...,))
    s12 = size_up_and_reconnect(S, Symbol(new_bond))
    
    # Rescale the intersecting dimension in U and s12
    common_id1 = intersect(inds(U), inds(s12))
    common_id2 = intersect(inds(Vt), inds(s12))

    # print("The characteristics of s12:", size(s12), inds(s12))
    # Cut all the relevant tensors towards the size specified by the bondsize argument
    s12 = tensor_cutting_index!(s12, Symbol(common_id1[1]), bondsize)
    s12 = tensor_cutting_index!(s12, Symbol(new_bond), bondsize)
    # here is the problem --> The network already has this index with this specific size associated with it... how to delete this from tensor information
    U = tensor_cutting_index!(U, Symbol(common_id1[1]), bondsize)
    s12 = tensor_cutting_index!(s12, Symbol(common_id2[1]), bondsize)
    s12 = tensor_cutting_index!(s12, Symbol(new_bond), bondsize)

    Vt = tensor_cutting_index!(Vt, Symbol(common_id2[1]), bondsize)

    # compute the new tensors
    A_new = contract(Q1,contract(U, s12))
    B_new = contract(Q2, contract(Vt, s12))

    if printing == true
        println("computing new A")
        println("sizes:",size(Q1), size(U), size(s12))
        println("indices:",inds(Q1), inds(U), inds(s12))
        println("A_new computed")
        println("Size difference:", size(tensor1), size(A_new))
        println("Index difference:", inds(tensor1), inds(A_new))
        
    end   

    if printing == true
        println("computing B new")
        println("sizes:",size(Q2), size(Vt), size(s12))
        println("indices:",inds(Q2), inds(Vt), inds(s12))
        println("B_new computed")
        println("Size difference:", size(tensor2), size(B_new))
        println("Index difference:", inds(tensor2), inds(B_new))
    end

    # Replacing the relevant things inside of the TN    
    pop!(tn, tensor1)
    push!(tn, A_new)
    pop!(tn, tensor2)
    push!(tn, B_new)
    replace!(tn, Symbol(new_bond) => index)   # Perform index renaming back to the original name --> If this insn't done, big error
    return tn

end

function grouping_bondindices(tn, indices_to_group, printing=false)
    """ Matricization of a tensor involves rearranging the data of a tensor into a matrix form based on a specific grouping of indices. 
    This function takes in a TensorNetwork and a list of indices insize this network which should be grouped together """

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
    if printing == true
        println("First replaced tensor sizes ", size(tensor1), "==>", size(tensor1_new), "where ", inds(tensor1), "==>", inds(tensor1_new))
        println("Second replaced tensor sizes ", size(tensor2), "==>", size(tensor2_new), "where ", inds(tensor2), "==>", inds(tensor2_new))
    end  

    # Modify the tesnor network by replacing the old tensors with the new tensors where they have a grouped bond
    pop!(tn, tensor1)
    push!(tn, tensor1_new)
    pop!(tn, tensor2)
    push!(tn, tensor2_new)
   

    indices_mapping = Dict{String, String}()  # Specify the type of the keys and values

    for index in indices_to_group
        indices_mapping[string(index)] = new_index
    end

    return indices_mapping
   
end

function contraction_step(tn, bond_index)
    """ Helper function to allow contraction of the tensor network along a specified index """
    if length(inds(tn)) == 0
        return
    end

    if string(bond_index) in [string(id) for id in inds(tn)]
        tensor1, tensor2 = Tenet.select(tn, bond_index)

        contracted = contract(tensor1, tensor2)
        # Replacing the relevant things inside of the TN    
        pop!(tn, tensor1)
        pop!(tn, tensor2)
        push!(tn, contracted)
        
    end



    
end


function approximate_contraction(tn, max_chi, list_of_contraction_indices, visualisation = false)
    """ Function which takes in a tensor network and computes its approximate_contraction based on a sequence of indices """
    
    for i in 1:length(tensors(tn))
        # Select both tensors connected to the index
        active_index = list_of_contraction_indices[i]
        tensor1, tensor2 = Tenet.select(tn, (Symbol(active_index)))

        # Check how many shared indices there are between the two tensors.
        shared_indices = intersect(inds(tensor1), inds(tensor2))
        println(shared_indices, length(shared_indices))

        if length(shared_indices) > 1
            index_map = grouping_bondindices(tn, shared_indices)
            for index in keys(index_map)
                replace!(list_ordered_contraction_indices, index => index_map[index])
                active_index = index_map[index]
            end
            # Modify the original list in place to remove duplicates -> some indices get mapped to the same grouped index, remove the duplicates
            unique!(list_ordered_contraction_indices)
        end


        # Check if the bondsize is larger than what we want -> if condition is met compress the bond
        if size(tn, Symbol(active_index)) > max_chi
            println("Performing truncated SVD to chi = $max_chi")
            truncated_SVD_replacement(tn, Symbol(active_index), max_chi)
        end

        # Contracting the network along the active index
        tensor1, tensor2 = Tenet.select(tn, Symbol(active_index))

        contracted = contract(tensor1, tensor2)
        # Replacing the relevant things inside of the TN    
        pop!(tn, tensor1)
        pop!(tn, tensor2)
        push!(tn, contracted)
        if length(tensors(tn)) == 1
            return (tensors(tn)[1][1])
        end
        # Visualisation of the contraction step

        if visualisation == true && length(tensors(tn)) > 1
            #println("generating plot")
            # Make a drawing of the tensor network
            # -> Takes a long long time for larger networks...
            drawing1 = plot(tn, labels=true)
            display(drawing1)
        end
        
    end
        
end


""" 

Sketch of how the approximate_contraction algorithm will work, it takes in a list of indices and a tensor network

-> Iterates over this list of indices and pops the first index:

1. Checks if this index is present in the network.
2. If this index is present, select the two tensors it connects to
3. Compute the intersecting indices between the two tensors
4. If they share more than 1 common index => group indices in the network and use the grouping dictionary to update the contraction sequence
5. Pop out the old contraction indices of the list of contract order
6. Before contracting the two tensors, check that the dimension of the grouped indices between the two tensors if less than the prespecified bondsize
7. If the size of the index to be contracted over is bigger than the size of the maximal bondsize, perform a truncated_SVD_replacement on this bond
8. Contract the two tensors

Repeat these steps for the whole list of contraction indices

"""


# Compute a greedy contraction sequence -> list of indices
contraction_sequence = EinExprs.einexpr(TN, optimizer=EinExprs.Greedy)
#list_ordered_contraction_indices_start = [string(index) for index in EinExprs.inds(contraction_sequence)]

exact_contraction_value = contract(TN)[1]
approximate_contraction_values = [Float32(0)]
chi_list = [Float32(0)]

for chi_max in 1:4
    TN_editable = copy(TN)
    global list_ordered_contraction_indices = [string(index) for index in EinExprs.inds(contraction_sequence)]
    approximate_contraction_value = approximate_contraction(TN_editable, chi_max, list_ordered_contraction_indices, true)
    relative_error = 1 - (approximate_contraction_value/exact_contraction_value)
    push!(approximate_contraction_values, Float32(relative_error))
    push!(chi_list, Float32(chi_max))


end


println(typeof(chi_list))
println(typeof(approximate_contraction_values))

plot(chi_list, approximate_contraction_values, xlabel="Chi", ylabel="Relative Error", title="Approximate Contraction Error vs. Chi")



























































initial_contraction = contract(TN)[1]
println("Value of initial network contraction = ", initial_contraction)




println("=================================================================")
println("Preparing the TensorNetwork for compressed contraction")
println("Performing bond grouping on the whole network")

# Matricization of the tensornetwork
for grouping in generate_groupedbonds(TN)
    if length(grouping) == 1
        continue
    else
        grouping_bondindices(TN, grouping)
    end
end

contract_after_grouping = contract(TN)[1]
println("Value of contraction after bondgrouping = ", contract_after_grouping)


if make_plot == true
    println("generating plot")
    # Make a drawing of the tensor network
    # -> Takes a long long time for larger networks...
    drawing = plot(TN, labels=true)
    display(drawing)
end


# Compute a greedy contraction sequence -> list of indices
contraction_sequence = EinExprs.einexpr(TN, optimizer=EinExprs.Greedy)
ordered_contraction_indices = EinExprs.inds(contraction_sequence)


#something something going wrong with the dimensions....


chi = 4
# Perform truncated SVD replaced on the whole network
for index in ordered_contraction_indices
    println(" Performing bond compression on index ")
    truncated_SVD_replacement(TN, index, chi, false)
    #println("truncated SVD on index  completed")
end

# Contracting the network along the ordered contraction sequence
# -> Problem when two bonds get folded together

for (i, index) in enumerate(ordered_contraction_indices)
    println("Performing contraction of index ")
    println(length(tensors(TN)))
    if length(tensors(TN)) == 1
        break
    end
    contraction_step(TN, index)
    if make_plot == true && length(tensors(TN)) > 1
        println("generating plot")
        # Make a drawing of the tensor network
        # -> Takes a long long time for larger networks...
        drawing1 = plot(TN, labels=true)
        display(drawing1)
    end
end


compressed_contraction_SVD = tensors(TN)[1][1]
println("Value of contraction after truncated_SVD_replacement = ", compressed_contraction_SVD)





valchi1 = 1.3905168314992278e12