# Add in the neccesary libraries and depedencies
# First precompile takes more time -> Julia creates a type structure so that it can exploit the JIT feature
using Plots
using Makie
using GraphMakie.NetworkLayout
using CairoMakie
using Tenet
using TensorOperations
using LinearAlgebra
using Graphs
using Colors
using GraphPlot
using EinExprs
using Combinatorics
using LaTeXStrings
Makie.inline!(true)


"""
Julia file with all the relevant functions which can be reused in different
codes by added them with the include("julia_functions.jl") command.
Make sure to copy this file to the correct directory beforehand.
"""


# This file will contain some nice and importable functions which allows one to work inside of a notebook without all cluttered workspace
# Main big functions -> these functions use some functionality from the smaller helper functions below
function truncated_SVD_replacement(tn, index, bondsize, printing=false)

    """ 
    Function which takes in a TensorNetwork and an index which connect two tensors inside of this network.
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
        println("Q1 Tensor = ", size(Q1), inds(Q1))
        println("R1 Tensor = ", size(R1), inds(R1))
        println("Q2 Tensor = ", size(Q2), inds(Q2))
        println("R2 Tensor = ", size(R2), inds(R2))
    end

    # Perform a SVD-decomposition on the R12 tensor -> truncation of the S - matrix will allow one to make an approximation to the network

    new_bond = string(index)*"_*"

    R12 = Tenet.contract(R1, R2)

    # Some-decission making based on the tensor R12
    # As the amount of singular values on the bond is given by the dimensions of the legs coming out of the R12 tensor and not by the size of the bond in the network....
    if bondsize > minimum([size(R12, ix) for ix in inds(R12)])
        return  # Exit the function prematurely and don't Perform the SVD-> truncation as the amount of singular values are to small compared to the bondsize
    end

    Q1_inds = intersect(inds(Q1), inds(R12)) #extract the indices pointing towards Q1
    U, S, Vt = LinearAlgebra.svd(R12, left_inds = (Q1_inds...,))
    s12 = size_up_and_reconnect(S, Symbol(new_bond))

    if printing == true
        println("R12 Tensor = ", size(R12), inds(R12))
        println("U Tensor = ", size(U), inds(U))
        println("S Tensor = ", size(S), inds(S))
        println("s12 Tensor = ", size(s12), inds(s12))
        println("Vt Tensor = ", size(Vt), inds(Vt))
    end

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
    A_new = Tenet.contract(Q1,Tenet.contract(U, s12))
    B_new = Tenet.contract(Q2, Tenet.contract(Vt, s12))

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
    return 

end

function grouping_bondindices(tn, indices_to_group, printing=false)

    """ 
    Matricization of a tensor involves rearranging the data of a tensor into a matrix form based on a specific grouping of indices.
    Matricization allows the contraction process to be better understood as pairwise contractions.
    This function takes in a TensorNetwork and a list of indices insize this network which should be grouped together
    """

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
   

    indices_mapping = Dict{Symbol, Symbol}()  # Specify the type of the keys and values

    for index in indices_to_group
        indices_mapping[Symbol(index)] = Symbol(new_index)
    end

    return indices_mapping
   
end

function approximate_contraction(tn, max_chi, list_of_contraction_indices, visualisation = false, svd_printing = false)

    """ 
    Function takes in a TensorNetwork, a maximal allowed bondsize and a sequence of contraction indices.
    It uses pairwise approximate contraction based on SVD truncation to max_chi on the bond following a contraction a sequence of indices.
    It generates a mapping of indices to new indices when the need for bondgrouping arrises and updates the contraction path when needed.
    """
    
    for i in 1:length(tensors(tn))
        # Select both tensors connected to the index
        active_index = list_of_contraction_indices[i]
        tensor1, tensor2 = Tenet.select(tn, (Symbol(active_index)))

        # Check how many shared indices there are between the two tensors.
        shared_indices = intersect(inds(tensor1), inds(tensor2))
        #println(shared_indices, length(shared_indices))

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
        #println("The size of the active index is = ", size(tn, Symbol(active_index)))
        if size(tn, Symbol(active_index)) > max_chi
            #println("Performing truncated SVD to chi = $max_chi")
            truncated_SVD_replacement(tn, Symbol(active_index), max_chi, svd_printing)
        end

        # Contracting the network along the active index
        tensor1, tensor2 = Tenet.select(tn, Symbol(active_index))

        contracted = Tenet.contract(tensor1, tensor2)
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
            drawing1 = Makie.plot(tn, labels=true)
            display(drawing1)
        end
        
    end
        
end

function TN2D_classical_ising_partition_function(Nx::Int, Ny::Int, beta, plotting=false)
    
    """ 
    Function which creates a ferromagnetic (j=1) classical ising model partition function on a square grid.
    Takes in the sizes of the IsingModel Nx, Ny and β.
    Function returns a Tenet TensorNetwork
    """

    # boltzman matrix
    B = [exp(beta) exp(-beta); exp(-beta) exp(beta)]
    sqrtB = convert(Matrix{Float64}, B^(1/2)) ### USE THE CONVERT BECAUSE JULIA HAS ALL THE TYPES OF OBJECTS? -> works
    # sqrtB = B^(1/2) # -> doesn't work -> Symetric? 

    # creating the building blocks of the ising tensor network
    # getting to know einsum notation in Jutho's tensor operations

    # Generating the relevant data arrays -> A2 = corner Tensor, A3 = edge Tensor, A4 = bulk Tensor
    @tensor begin
        A2[i,j] := sqrtB[i,k]*sqrtB[k,j] #contraction along k
        
        ### JULIA HAS INDICES STARTING AT ONE !!!!!!!!!!!!!!!!!!!!!!! STOP FORGETTING
        A3[i,j,k] := sqrtB[:,1][i]*sqrtB[:,1][j]*sqrtB[:,1][k] + sqrtB[:,2][i]*sqrtB[:,2][j]*sqrtB[:,2][k] #is this the easiest way to do an outer product?
        A4[i,j,k,l] := sqrtB[:,1][i]*sqrtB[:,1][j]*sqrtB[:,1][k]*sqrtB[:,1][l] + sqrtB[:,2][i]*sqrtB[:,2][j]*sqrtB[:,2][k]*sqrtB[:,2][l]

    end 


    tensors = []

    ## the layout structure --> Use the Graphs.jl package for underlying functionality... Don't code it all yourself

    n_tensors = Nx*Ny
    n_edges = 2*Nx*Ny - Nx - Ny
    G = Graphs.grid([Nx,Ny])
    nvertices = nv(G) # number of vertices
    nedges = ne(G)    # number of edges

    nodes = [node for node in vertices(G)]
    nodes_labels = [[i,j] for i in 1:Nx for j in 1:Ny]
    nodes_map = Dict(zip(nodes, nodes_labels)) # Create a mapping dict
    edgesgraph = [edge for edge in edges(G)]
    edges_labels = [Symbol(edge) for edge in 1:nedges]
    edges_map = Dict(zip(edgesgraph, edges_labels))

    if plotting == true
        display(gplot(G, nodelabel=nodes, edgelabel=1:n_edges, layout=spectral_layout))
    end


    
    # Generating the tensors inside of the network
    for source in vertices(G)
        inds = Tuple([edges_map[edge] for edge in edges(G) if source in([src(edge), dst(edge)])])
        if length(inds) == 2
            push!(tensors, Tensor(A2, inds))
        end
        if length(inds) == 3
            push!(tensors, Tensor(A3, inds))
        end
        if length(inds) == 4
            push!(tensors, Tensor(A4, inds))
        end   
        
    end

    ising_network = TensorNetwork(tensors)
    return ising_network

    # Generate a TensorNetwork based on all the tensors in the list of tensors
    # Return this as a Tenet tensor network object
end


function generate_all_possible_looptotree_replacements(n)

    """ 
    This function takes in a number n which represents the amount of vertices in the loop which we want to replace.
    The is done purely on graph-level reasoning, the fact that the graph represents
    a complex network of interconnected tensors is not yet taking into account.
    The function returns a list of simple graphs which are the possible 
    spanning tree replacements for this tensor loop.
    """

    # Generate a complete graph of order n to extract all possible spanning trees on this graph structure
    G = Graphs.complete_graph(n)
    node_coloring = [colorant"orangered2", colorant"cyan", colorant"chartreuse", colorant"goldenrod1", colorant"purple", colorant"lightskyblue4
    ", colorant"mediumpurple1"][1:nv(G)]
    nodes = [node for node in vertices(G)]      
    number_of_spanning_trees = n^(n-2)
    # create a masterlist which is the set of all edges represented as a sorted (Read undirected) tuple of vertices.
    masterlist = [Tuple(sort([src(edge), dst(edge)])) for edge in edges(G)] 


    # Generate the list of all edges which possibly could be a tree and filter out the ones which are not  a tree
    tree_like = generate_combinations(masterlist, length(nodes)-1)

    # Generate all possible spanning trees by checking if their is no loop in the tree_like edge list
    all_spanning_trees = [tree for tree in tree_like if graph_is_tree(tree, nodes) == true]
 
    # Creating a list of graphs objects for all the trees
    spanning_tree_graphs = []
    for spanningtree_edges in all_spanning_trees
        # Example usage
        g  = SimpleGraph(n)
        for edge in spanningtree_edges
            add_edge!(g, edge[1], edge[2])
        end
        if is_tree(g) == true # extra condition for the n = 5 case -> my graph_is_tree_function doesn't catch the disconnected subloops.
            push!(spanning_tree_graphs, g)
        end
    end


    return spanning_tree_graphs
end

# Smaller helper functions
function contraction_step(tn, bond_index)

    """ 
    Helper function to allow contraction of the tensor network
    along a specified index 
    """
    
    if length(inds(tn)) == 0
        return
    end

    tensor1, tensor2 = Tenet.select(tn, bond_index)

    contracted = Tenet.contract(tensor1, tensor2)
    # Replacing the relevant things inside of the TN    
    pop!(tn, tensor1)
    pop!(tn, tensor2)
    push!(tn, contracted)
        
    

end


function generate_groupedbonds(tn)

    """ 
    Generates a dictionary containing as keys lists of bond-indices
    between two tensors and as vals the tensors in connecting the bonds
    """

    result_dict = Dict{Vector, Vector}()

    for id in inds(tn)
        tensor1, tensor2 = Tenet.select(tn, (Symbol(id)))
        common_indices = intersect(inds(tensor1), inds(tensor2))
        result_dict[common_indices] = [tensor1, tensor2]
    end

    return keys(result_dict)

end

function size_up_and_reconnect(S, index)

    """ 
    This helper function takes in tensor S from the SVD and splits it 
    in two -> two times s^(1/2) reconnects them with the index of the bond 
    """

    s12_data = S.data.^(1/2)
    s12_data_matrix = LinearAlgebra.diagm(s12_data)
    s12_indices = [inds(S)..., Symbol(index)]
    s12_tensor = Tensor(s12_data_matrix, (s12_indices...,))

    return s12_tensor

end

function tensor_cutting_index!(tensor, index, bondsize)

    """ 
    Function which takes in a tensor, a symbolic index and a bondsize. 
    Reduces the dimension of this index to bondsize 
    """
    
    slice_location = dim(tensor, index)
    # Create a tuple of slices for each dimension
    slices = [1:size(tensor, dim) for dim in 1:ndims(tensor)]
    # Replace the slice for the specified dimension with the desired range
    slices[slice_location] = 1:bondsize
    # Create a new tensor with the same indices, but the data is rescaled along one dimension
    tensor = Tensor(tensor[slices...], inds(tensor))
    return tensor

end


function graph_is_tree(edge_list, vertices_list)

    """ 
    Helper function which checks is a list of edges is a tree based on a given vertices_list
    This is done by checking if there are nv(G)-1 edges in this proposed graph
    The connectivity condition is evaluated as well
    """

    # Check the nv(G)- 1 edges condition
    if length(edge_list) != length(vertices_list)-1
        return false
    end

    sort!(vertices_list) # sort the vertices_list for comparisson purposes

    # Extract unique elements from the tuples
    unique_elements = Set(v for tpl in edge_list for v in tpl)
    # Convert the Set to a sorted list for comparisson
    unique_elements_list = sort(collect(unique_elements))
    
    print(vertices_list)
    # Check the connectivity condition
    if unique_elements_list == vertices_list 
        return true
    end

    return false 

end


function generate_combinations(full_edge_list, length)

    """
    Helper function created using the Combinatorics.jl package.
    Based on a input list of edges this function generates all possible combinations of a certain length
    If length is set to nv(G)-1 this generates all possible edge lists which could be trees
    """
    
    # Create a list to store all the combinations
    all_combinations = [combo for combo in combinations(full_edge_list, length)]

    return all_combinations
end


function fill_with_random(G, dims, visualisation = false, fixed_dim = true)

    """
    Function which takes in a Graphs.jl graph-structure (G) and which generates a Tenet.TensorNetwork based on the connectivity of the network.
    First of all each node is filled with a random tensor, the dims arguement shows the size of each index.
    If the dims argument is passed an [lowest_dim, highest_dim] argument each index size is randomly choosen from this range 
    """
    
    nvertices = nv(G) # number of vertices
    nedges = ne(G)    # number of edges

    nodes = [node for node in vertices(G)]
    edgesgraph = [edge for edge in edges(G)]
    edges_labels = [Symbol(edge) for edge in 1:nedges]
    edges_map = Dict(zip(edgesgraph, edges_labels))

    if visualisation == true
        display(gplot(G, nodelabel=nodes, nodefillc=colorant"springgreen3", layout=spring_layout))
    end


    tensors = []
    # Generating the tensors inside of the network
    for source in vertices(G)
        inds = Tuple([edges_map[edge] for edge in edges(G) if source in([src(edge), dst(edge)])])
        if fixed_dim == true
            size_generation_tuple = Tuple([dims for i in 1:length(inds)])
        end
        if fixed_dim == false
            size_generation_tuple = Tuple([rand(dims[1]:dims[2]) for i in 1:length(inds)])
        end
        push!(tensors, Tensor(rand(size_generation_tuple...), inds))
        
    end

    TN = TensorNetwork(tensors)
    return TN

end

function extract_graph_representation(TN, printing=false)

    """
    Function which takes in a TensorNetwork from Tenet.jl, this network only has one edge between two tensors (grouped indices) and no self loops.
    Based on the connectivity inside of the TensorNetwork a simple graph structure is generated.
    For this mapping a dictionary which has vertex labels as keys and corresponding tensors as values is generated.
    For this mapping a dictionary which has the corresponding edge as keys and as values the corresponding tensors indices.
    A fully weighted edge list is generated [[source, drain, size], ...]
    An edge to index map is generated Dict[(source, drain)] -> Tensor network index
    """

    n_vertices = length(tensors(TN))
    n_edges =  [[inds(tensor)...] for tensor in tensors(TN)]
    to_set = []
    for group in n_edges
        for element in group
            push!(to_set, element)
        end
    end
    n_edges = length(Set(to_set))
    
    if printing == true
        println("Amount of extracted vertices = ", n_vertices, "\nAmount of extracted edges = ", n_edges)
    end

    list_of_edges = collect((Set(to_set)))
    tensor_vertex_map = Dict{Int, Tenet.Tensor}()  # Specify the type of the keys and values
    for (i, tensor) in enumerate(tensors(TN))
        tensor_vertex_map[Int(i)] = tensor
    end

    if printing == true
        println(tensor_vertex_map)
    end

    g = SimpleGraph(n_vertices)
    nodes = [node for node in vertices(g)]

    # the connectivty inside of the tensor network should be mapped onto the connectivty of the SimpleGraph
    index_edge_map = Dict{}()

    fully_weighted_edge_list = []                                               # generate a list containing the weighted edges [source, drain, size]
    edge_index_map = Dict{}()                                                   # dictionary which maps (source, drain) tuples to tensor network indices

    pairs = collect(combinations([node for node in vertices(g)], 2))
    for possible_connection in pairs
        v1 = possible_connection[1]
        v2 = possible_connection[2]
        T_v1 = tensor_vertex_map[v1]
        T_v2 = tensor_vertex_map[v2]
        # connectivity measure
        index_intersection =  intersect(inds(T_v1), inds(T_v2))
        if length(index_intersection) == 1
            add_edge!(g, v1, v2)
            edge = [edge for edge in edges(g) if [src(edge), dst(edge)] == possible_connection][1]
            index_edge_map[index_intersection] = edge
            push!(fully_weighted_edge_list, (v1, v2, size(TN, index_intersection[1])))
            edge_index_map[(v1,v2)] = index_intersection
        end
    end

    if printing == true
        display(gplot(g, nodelabel=[node for node in vertices(g)]))
        println(index_edge_map)
    end

    return g, tensor_vertex_map, index_edge_map, fully_weighted_edge_list, edge_index_map

end



function sized_adj_from_weightededges(fully_weighted_edge_list, graph)

    """
    Function which takes in the fully weighted edge list [[source, drain, size], ...]
    and the current graph representation in gives the corrseponding sized adj_matrix.
    """

    num_vertices = nv(graph)

    # Initialize a sized adjacency matrix with zeros
    sized_adj_matrix = zeros(Int, num_vertices, num_vertices)

    # Add edges to the adjacency matrix with specified sizes
    for weighted_edge in fully_weighted_edge_list
        u = weighted_edge[1]
        v = weighted_edge[2]
        size = weighted_edge[3]
        # Modify the matrix
        sized_adj_matrix[u, v] = size
        sized_adj_matrix[v, u] = size  # Add reverse edge to ensure symmetry
    end

    return sized_adj_matrix
end 

function cycle_basis_to_edges(cycle_basis)

    """
    This function takes in a cycle basis of a graph and return
    a list of sorted tuples containing all edges inside of the graph
    """

    edges = []

    for cycle in cycle_basis
        # extract the edges from the current cycle
        c_edges = [[cycle[i], cycle[i+1]] for i in 1:length(cycle)-1]
        push!(c_edges, [cycle[end], cycle[1]])
        # add the sorted edge to the edges if not already inside of edges
        for edge in c_edges
            if Tuple(sort(edge)) in edges                                              
                continue
            end
            push!(edges, Tuple(sort(edge)))
        end
    end
   
    return sort(edges)
end



function update_edge_availability(graph, fully_weighted_edge_list, boolean_edge_availability)

    """
    Function which takes in a freshly modified Graphs.jl structure, 
    a fully_weighted_edge_list, a current boolean_edge_availability list and
    this functions updates the boolean_edge_availability list based on the
    current graph structure (after removing an edge)
    """

    # Get the cycle_basis of the graph which represent all the simple cycles
    cycles = cycle_basis(graph)
    cycle_edges = cycle_basis_to_edges(cycles)                                  # extract a list of the possible available [source, drain] combinations

    # For each list in the fully_weighted_edge_list 
    # update the boolean_edge_availability list based on if this specific edge
    # is inside of the cycle_edges

    for (i, edge_weight) in enumerate(fully_weighted_edge_list)
        edge = edge_weight[1:2]
        if edge ∈ cycle_edges
            boolean_edge_availability[i] = true
        else
            boolean_edge_availability[i] = false
        end
    end
    
    return boolean_edge_availability

end


function extract_edge_representation_and_physical_indices(graph, cycle)

    """
    Helper function which probes the graph structure given a cycle.
    The function extract all the edges and all the dangling edges
    if this loop would be cut out off the graph.
    """
    
    # First extract the cycle edges 
    cycle_edges = []
    # Extract the edges from the current cycle
    c_edges = [[cycle[i], cycle[i+1]] for i in 1:length(cycle)-1]
    push!(c_edges, [cycle[end], cycle[1]])
    # add the sorted edge to the edges if not already inside of edges
    for edge in c_edges
        if Tuple(sort(edge)) in cycle_edges                                              
            continue
        end
        push!(cycle_edges, Tuple(sort(edge)))
    end 


    # Now extract the dangling edges
    # List of all edges connected to cycle vertices -> filter out cycle edges
    dangling = []
    for vertex in cycle
        for connected_vertex in neighbors(graph, vertex)
            if Tuple(sort([vertex, connected_vertex])) in cycle_edges
                continue
            end
            push!(dangling, Tuple(sort([vertex, connected_vertex])))
        end
    end

    return sort(cycle_edges), sort(dangling)

end


function calculate_DMRG_cost(graph, weighted_edges, selected_cycle, selected_edge)

    """
    Initial cost function which takes in a graph, list of edges with their 
    weights, the selected cycle and the selected edge in this cycle.
    Based on this information it calculates the DMRG cost in the χ^5 cost model
    """

    loop_active, dang_active = extract_edge_representation_and_physical_indices(graph, selected_cycle)

    # Get a list of all participating bond dimensions in the MPS
    # Extract the maximal occuring bond dimensions
    # Cost model it as Χ^5

    chi_in_choosen_MPS = []
    for edge in vcat(loop_active, dang_active)
        if edge == selected_edge
            continue
        else
            for (v1,v2,w3) in weighted_edges
                if Tuple(sort([v1,v2])) == edge
                    push!(chi_in_choosen_MPS, w3)
                    break
                end
            end
        end
    end

    return maximum(chi_in_choosen_MPS)^5
end


function display_selected_cycle(graph, cycle)

    """
    This function takes in a graph from graphs.jl, a cycle from the 
    cycle basis of the graph. By generating a graphplot with a 
    color code the selected cycle is visualized.
    """

    nodes = [node for node in vertices(graph)]
    loop_active, dang_active = extract_edge_representation_and_physical_indices(graph, cycle)
    colors_for_edges = []
    for edge in edges(graph)
        s = src(edge)
        d = dst(edge)
        
        if Tuple((Int(s), Int(d))) in loop_active
            push!(colors_for_edges, colorant"seagreen2")
            continue
        end
        
        push!(colors_for_edges, colorant"grey")
    end
    
    # Planar embedding for the Frucht Graph example
    #TODO: Generalize to given the embedding as arguments and if not given
    # automatically resort to spring layout inside of GraphPlot.jl
    locs_x =     [4, 4, -5, -2, 0, 0, 2, 0, -3, -1, -6, -4]
    locs_y =  -1*[-2, 1, -2, -1, 0, -2, 0, 3, 3, 1, 1, 0]

    display(gplot(graph, locs_x, locs_y, edgestrokec = colors_for_edges, nodelabel=nodes, nodefillc=colorant"springgreen3"))
    
end


function display_selected_action(graph, cycle, choosen_edge)

    """
    This function takes in a graph from graphs.jl, 
    a cycle inside from the cycle basis of the graph and a choosen edge
    on this graph which is the where the cut is made.
    By generating a graphplot with a color code the selected edge and underlying
    MPS structure for appling DMRG is visualized.
    """

    nodes = [node for node in vertices(graph)]
    loop_active, dang_active = extract_edge_representation_and_physical_indices(graph, cycle)
    colors_for_edges = []
    for edge in edges(graph)
        s = src(edge)
        d = dst(edge)
        if Tuple((Int(s), Int(d))) == choosen_edge
            push!(colors_for_edges, colorant"green2")
            continue
        end
        if Tuple((Int(s), Int(d))) in loop_active
            push!(colors_for_edges, colorant"seagreen2")
            continue
        end
        if Tuple((Int(s), Int(d))) in dang_active
            push!(colors_for_edges, colorant"darkolivegreen")
            continue
        end
        
        push!(colors_for_edges, colorant"grey")
      
    end

    # Planar embedding for Frucht Graph example
    locs_x =     [4, 4, -5, -2, 0, 0, 2, 0, -3, -1, -6, -4]
    locs_y = -1*[-2, 1, -2, -1, 0, -2, 0, 3, 3, 1, 1, 0]

    display(gplot(graph, locs_x, locs_y, edgestrokec = colors_for_edges, nodelabel=nodes, nodefillc=colorant"springgreen3"))
    
end


function minimum_cycle_basis(graph)

    """
    Return a minimum cycle basis / inner faces of a planar graph.
    This function lists all fundamental loops which are present on the graph,
    this in graph theoretical perspective comes out to finding the minimum
    cycle basis instead of a general cycle basis (of which multiple exist)
    """

    cycle_basisses = []
    
    for root in vertices(graph)
        for cycle in cycle_basis(graph, root)
            if cycle ∉ cycle_basisses
                push!(cycle_basisses, cycle)
            end
        end
    end

    unique_repr = []
    unique_faces = []
    for face in cycle_basisses
        sorted_face = Tuple(sort(face))
        if !(sorted_face in unique_repr)
            push!(unique_repr, sorted_face)
            push!(unique_faces, face)
        end
    end

    # Sort lists by their lengths
    sorted_faces = sort(unique_faces, by=length)

    # Select the smallest possible elements forming a cycle basis
    smallest_loops = sorted_faces[1:length(cycle_basis(graph))]
    return smallest_loops
end


function create_actionmatrix(graph, all_edges)

    """
    Function which takes in a graphs.jl graphs object and which generates the
    possible actions in a matrix format. Here the collums represent possible 
    edges and here the rows identify with the different possible faces. 
    This way selecting a element of action matrix A[i,j] corresponds to
    selecting both a possible face and an edge on the circumfrence of the face.
    """

    faces = minimum_cycle_basis(graph)
    edges_graph = all_edges
    edge_basis = cyclebasis_to_edgebasis(faces)

    # Create an action matrix with dimensions of #cycles ₓ #edges
    A = zeros(Int, length(edge_basis), length(edges_graph))
        
    # Iterate through each cycle and check if each edge is inside
    for (i, edgegroup) in enumerate(edge_basis)
        for (j, edge) in enumerate(edges_graph)
            if any(c == edge for c in edgegroup)
                A[i, j] = 1
            end
        end
    end
    return A
end


function cyclebasis_to_edgebasis(minimal_cyclebasis)

    """
    Helper function for generating the possible edges inside of cycle,
    this allows to easily generate the action matrix.
    """
    
    edge_basis = []

    for cycle in minimal_cyclebasis
        # extract the edges from the current cycle
        c_edges = [Tuple(sort([cycle[i], cycle[i+1]])) for i in 1:length(cycle)-1]
        push!(c_edges, Tuple(sort([cycle[end], cycle[1]])))
        push!(edge_basis, c_edges)
    end
    return edge_basis
end


function edge_weights_update_DMRG_exact(old_graph, selected_cycle, selected_edge, weighted_edge_list)
    
    """
    Exact DMRG dimensionality updating function. This function takes in the old
    graph structure, the selected_cycle as a list of vertices, the selected edge
    as a tuple and the current weighted_edge_list.

    Based on the possible information flow through the dimensions of the tensor
    network it computes the updated edge weights and returns a new updated
    weighted_edge_list.
    """

    # extract edges inside of the loop and the ones which are dangling from the loop
    loop_active, dang_active = extract_edge_representation_and_physical_indices(old_graph, selected_cycle)
    virtual_MPS_edges = filter!(x -> x != selected_edge, loop_active)           # filter out the choosen edge which is cut 


    # create a dictionary for easely accesing the dimensions of the edges inside of the network
    weights_dict = Dict((edge1, edge2) => weight for (edge1, edge2, weight) in weighted_edge_list)
    old_virtual_weights = [weights_dict[virtual_edge] for virtual_edge in virtual_MPS_edges]

    # Note: Important in this whole ordeal in having the correct ordering of edges and danling edges
    # This allows one to compute the correct dimensions inside of the exact DMRG replacement

    # Extract edge sequence based on walking along the cycle from 
    # selected_edge[1] towards selected_edge[2] 
    sd_virtual_edges_cycle = []

    # Extract dangling edges sequence when walking along the cycle from
    # selected_edge[1] towards selected_edge[2]
    sd_dangling_edges_cycle = []

    # walk along the cycle from source to drain
    current_position = selected_edge[1]
    end_vertex = selected_edge[2]
    
    # LOGIC FOR EXTRACTING ALL NEEDED EDGES IN THE RIGHT ORDERING SO THAT LEFT
    # AND RIGHT PRODUCT CAN BE DEFINED
    loop_length = length(virtual_MPS_edges)


    # walk the loop
    for i in 1:loop_length
        chosen_edge = [Tuple(sort([source, drain])) for (source, drain) in virtual_MPS_edges if source == current_position || drain == current_position][1]
        push!(sd_virtual_edges_cycle, chosen_edge)
        filter!(x -> x != chosen_edge, virtual_MPS_edges)

        dangling_edge = [Tuple(sort([source, drain])) for (source, drain) in dang_active if source == current_position || drain == current_position]
        # can sometimes not find such a tuple... 
        if length(dangling_edge) == 1
        # If it's not empty, get the second tuple (index 1) and sort it
            dangling_edge = dangling_edge[1]
        else
            # If it's empty, handle the case when no matching tuple is found
            dangling_edge = nothing                                             # match nothing with insert a size 1 product into the DMRG product replacement
        end

        push!(sd_dangling_edges_cycle, dangling_edge)
        
        # walk along the path
        current_position = (current_position == chosen_edge[1]) ? chosen_edge[2] : chosen_edge[1]


        # final DANLING EDGE? --> check at the end of the MPS structure
        if i == loop_length
            dangling_edge = [Tuple(sort([source, drain])) for (source, drain) in dang_active if source == current_position || drain == current_position]
            # can sometimes not find such a tuple... 
            if length(dangling_edge) == 1
            # If it's not empty, get the second tuple (index 1) and sort it
                dangling_edge = dangling_edge[1]
            else
                # If it's empty, handle the case when no matching tuple is found
                dangling_edge = nothing                                             # match nothing with insert a size 1 product into the DMRG product replacement
            end
            push!(sd_dangling_edges_cycle, dangling_edge)
        end
    end

    # usage of the ternary operator in julia -> julia-like-code
    sd_dangling_weights = [dangling_edge == nothing ? 1 : weights_dict[dangling_edge] for dangling_edge in sd_dangling_edges_cycle]
    new_virtual_weights = []

    # extract the new dimensions in the exact MPS representation based on
    # correct placement along the cycle

    for (i,edge) in enumerate(sd_virtual_edges_cycle)
        # println("right dangling =", prod(sd_dangling_weights[1:i]), " Left dangling = ", prod(sd_dangling_weights[i+1:end]))
        # push the minimum of [( product of left dangling dimensions ), (product of right dangling dimensions)]
        virtual_bound = minimum(old_virtual_weights)                                        # edge case if statement to make sure that no edge_weights are set to 1 by mistake min(1, prod(all)) shouldn't quench the bond.

        push!(new_virtual_weights,(edge[1], edge[2], max(min(prod(sd_dangling_weights[1:i]), prod(sd_dangling_weights[i+1:end])), virtual_bound)))
    end
    
    
    # Create a dictionary to store the new weights by (source, drain) pair
    new_weights_dict = Dict((source, drain) => weight for (source, drain, weight) in new_virtual_weights)

    # Generate the new list with updated weights
    new_weighted_edge_list = [(source, drain, haskey(new_weights_dict, (source, drain)) ? new_weights_dict[(source, drain)] : weight) for (source, drain, weight) in weighted_edge_list]
    
    return new_weighted_edge_list


end


function edge_weights_update_DRMG_chi_max(old_graph, selected_cycle, selected_edge, weighted_edge_list, chi_max)

    """
    Approximate DMRG dimensionality updating, this allows one to specify a 
    chi_max --> maximal allowable bond dimension in the virtual_MPS_edges. 
    This function takes in the old graph structure, the selected_cycle as a 
    list of vertices, the selected edge as a tuple and the 
    current weighted_edge_list.

    Based on the possible information flow through the dimensions of the tensor
    network it computes the updated edge weights and returns a new updated
    weighted_edge_list.
    """

    # extract edges inside of the loop and the ones which are dangling from the loop
    loop_active, dang_active = extract_edge_representation_and_physical_indices(old_graph, selected_cycle)
    virtual_MPS_edges = filter!(x -> x != selected_edge, loop_active)           # filter out the choosen edge which is cut 


    # create a dictionary for easely accesing the dimensions of the edges inside of the network
    weights_dict = Dict((edge1, edge2) => weight for (edge1, edge2, weight) in weighted_edge_list)
    old_virtual_weights = [weights_dict[virtual_edge] for virtual_edge in virtual_MPS_edges]

    # Note: Important in this whole ordeal in having the correct ordering of edges and danling edges
    # This allows one to compute the correct dimensions inside of the exact DMRG replacement

    # Extract edge sequence based on walking along the cycle from 
    # selected_edge[1] towards selected_edge[2] 
    sd_virtual_edges_cycle = []

    # Extract dangling edges sequence when walking along the cycle from
    # selected_edge[1] towards selected_edge[2]
    sd_dangling_edges_cycle = []

    # walk along the cycle from source to drain
    current_position = selected_edge[1]
    end_vertex = selected_edge[2]
    
    # LOGIC FOR EXTRACTING ALL NEEDED EDGES IN THE RIGHT ORDERING SO THAT LEFT
    # AND RIGHT PRODUCT CAN BE DEFINED
    loop_length = length(virtual_MPS_edges)


    # walk the loop
    for i in 1:loop_length
        chosen_edge = [Tuple(sort([source, drain])) for (source, drain) in virtual_MPS_edges if source == current_position || drain == current_position][1]
        push!(sd_virtual_edges_cycle, chosen_edge)
        filter!(x -> x != chosen_edge, virtual_MPS_edges)

        dangling_edge = [Tuple(sort([source, drain])) for (source, drain) in dang_active if source == current_position || drain == current_position]
        # can sometimes not find such a tuple... 
        if length(dangling_edge) == 1
        # If it's not empty, get the second tuple (index 1) and sort it
            dangling_edge = dangling_edge[1]
        else
            # If it's empty, handle the case when no matching tuple is found
            dangling_edge = nothing                                             # match nothing with insert a size 1 product into the DMRG product replacement
        end

        push!(sd_dangling_edges_cycle, dangling_edge)
        
        # walk along the path
        current_position = (current_position == chosen_edge[1]) ? chosen_edge[2] : chosen_edge[1]


        # final DANLING EDGE? --> check at the end of the MPS structure
        if i == loop_length
            dangling_edge = [Tuple(sort([source, drain])) for (source, drain) in dang_active if source == current_position || drain == current_position]
            # can sometimes not find such a tuple... 
            if length(dangling_edge) == 1
            # If it's not empty, get the second tuple (index 1) and sort it
                dangling_edge = dangling_edge[1]
            else
                # If it's empty, handle the case when no matching tuple is found
                dangling_edge = nothing                                             # match nothing with insert a size 1 product into the DMRG product replacement
            end
            push!(sd_dangling_edges_cycle, dangling_edge)
        end
    end

    # usage of the ternary operator in julia -> julia-like-code
    sd_dangling_weights = [dangling_edge == nothing ? 1 : weights_dict[dangling_edge] for dangling_edge in sd_dangling_edges_cycle]

    new_virtual_weights = []

    # extract the new dimensions in the exact MPS representation based on
    # correct placement along the cycle

    for (i,edge) in enumerate(sd_virtual_edges_cycle)
        # println("right dangling =", prod(sd_dangling_weights[1:i]), " Left dangling = ", prod(sd_dangling_weights[i+1:end]))
        virtual_bound = minimum(old_virtual_weights)                                        # edge case if statement to make sure that no edge_weights are set to 1 by mistake min(1, prod(all)) shouldn't quench the bond.

        push!(new_virtual_weights,(edge[1], edge[2], max(min(prod(sd_dangling_weights[1:i]), prod(sd_dangling_weights[i+1:end]), chi_max), virtual_bound)))
    end
    
    
    # Create a dictionary to store the new weights by (source, drain) pair
    new_weights_dict = Dict((source, drain) => weight for (source, drain, weight) in new_virtual_weights)

    # Generate the new list with updated weights
    new_weighted_edge_list = [(source, drain, haskey(new_weights_dict, (source, drain)) ? new_weights_dict[(source, drain)] : weight) for (source, drain, weight) in weighted_edge_list]
    
    return new_weighted_edge_list

end


function zero_padding_to_size(action_vector, size)

    d = size - length(action_vector)

    if d == 0
        return action_vector
    else
        for i in 1:d
            push!(action_vector, 0)
        end
    end

    return action_vector
end