

function get_edges_ary(nodes_ary::Array{Int64, 3})
    con_ary = nodes_ary[:, :, 1]
    sizes_ary = nodes_ary[:, :, 2]
    
    edges_ary = zeros(Int, size(con_ary))
    edges_ary[con_ary .>= 0] .= -2
    edges_ary[con_ary .== -1] .= -1
  
    num_edges = 1
    
    for i_nodes in sort(collect(enumerate(eachrow(con_ary))), rev =true)
        nodes = i_nodes[2]
        i = i_nodes[1]  
    
        for (j, node) in enumerate(nodes) 
            edge_i = edges_ary[i, j]
        
            if edge_i == -2
                nodes_ary_node = con_ary[node,:]
                _j = findfirst(x -> x == i, nodes_ary_node)
                edges_ary[i, j] = num_edges
                edges_ary[node, _j] = num_edges
                num_edges += 1
               
            end
        end
    end
  
    _edges_ary = maximum(edges_ary) .- edges_ary .+1
    _edges_ary[edges_ary .== -1] .= -1
    edges_ary = _edges_ary
    
    return edges_ary

end

function get_node_dims_arys(nodes_ary::Array{Int64, 3})
    num_nodes = size(nodes_ary,1)

    arys = Vector{Int}[]
    for (i, nodes) in enumerate(eachrow(nodes_ary[:, :, 1]))
        ary = zeros(Int, num_nodes) 
        for (j, node) in enumerate(nodes)
            positive_nodes = nodes[nodes .>= 0]
            if nodes_ary[i, j, 1] >0
            ary[node] =  nodes_ary[i, j, 2] 
            end
            end
        push!(arys, ary)
        
    end

    return arys
end



function get_node_bool_arys(nodes_ary::Array{Int64, 3})
    num_nodes = size(nodes_ary,1)

    arys = Vector{Bool}[]
    for (i, nodes) in enumerate(eachrow(nodes_ary[:, :, 1]))
        ary = zeros(Bool, num_nodes) 
        ary[i] = true
        
        push!(arys, ary)
    end
    
    return arys
end


#function update(edge::Int, edges_ary, node_dims_arys::Vector{Vector{Int}}, node_bool_arys::Vector{Vector{Bool}})
#    node_i0, node_i1 = first.(Tuple.(findall( x -> x == edge, edges_ary)))
#
#    n0 = node_dims_arys[node_i0]
#    n1 = node_dims_arys[node_i1]
#
#    n0_B = node_bool_arys[node_i0]
#    n1_B = node_bool_arys[node_i1]
#
#    if_diff = !(n0_B[node_i1])
#    if if_diff
#        
#        contract_dims = n0 .+ n1
#        contract_dims[n0.*n1 .!= 0 ] .=n0[n0.*n1 .!= 0 ] .* n1[n0.*n1 .!= 0 ]
#        
#        contract_bool = n0_B .| n1_B
#
#        flop = prod(filter(x -> x != 0, n0[.!contract_bool]))*prod(filter(x -> x != 0, n1[.!contract_bool]))*prod(filter(x -> x != 0, n0[contract_bool]))
#        
#        
#        contract_dims[contract_bool] .= 0  
#        
#        node_dims_arys[contract_bool] .= [contract_dims]
#        node_bool_arys[contract_bool] .= [contract_bool]
#
#        global flops += flop
#    end
#
#    
#
#end
#
#flops = 0
#nodes_ary = [2 5 -1; 1 3 5; 2 4 -1; 3 5 -1; 1 2 4;;; 4 4 -1; 4 4 4; 4 4 -1; 4 4 -1; 4 4 4]
#edges_ary = get_edges_ary(nodes_ary)
#print(typeof(edges_ary))
#print(edges_ary)
#print("\n ----------- \n")
#node_bool_arys = get_node_bool_arys(nodes_ary)
#
#node_dims_arys = get_node_dims_arys(nodes_ary)
#board =  cat(node_bool_arys, node_dims_arys, dims=3)
#print("\n ----------- \n")
#print("\n ----------- \n")
#print(board[:, :, 1][1])
#print(typeof(board))
#print("\n ----------- \n")
#print("\n ----------- \n")
#
#print(node_dims_arys, flops)
#print("\n ----------- \n")
#update(1, edges_ary, node_dims_arys, node_bool_arys )
#print(node_dims_arys, node_bool_arys, flops)
#print("\n ----------- \n")
#update(6, edges_ary, node_dims_arys, node_bool_arys )
#print(node_dims_arys, node_bool_arys, flops)
#print("\n ----------- \n")
#update(5, edges_ary, node_dims_arys, node_bool_arys )
#print(node_dims_arys, node_bool_arys, flops)
#print("\n ----------- \n")
#print("end")