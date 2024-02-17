using AlphaZero
using StaticArrays
import AlphaZero.GI
using Base: @kwdef


"""
Setup as GameEnv.
In this environment the state is a adjecancy matrix. The edges  
contain the EinExpr struct.
"""

include("functions.jl")


struct GameSpec <: GI.AbstractGameSpec end

mutable struct GameEnv <: GI.AbstractGameEnv
    edges::Matrix{Int64}
    node_dims_arys::Vector{Vector{Int64}}
    node_bool_arys::Vector{Vector{Bool}}
    reward_list::Array{Int64}
    amask::BitVector
    finished::Bool
    history :: Union{Nothing, Vector{Int}}
    
end


GI.spec(::GameEnv) = GameSpec()

function GI.init(::GameSpec)
    nodes_ary = [2 5 -1; 1 3 5; 2 4 -1; 3 5 -1; 1 2 4;;; 4 4 -1; 4 4 4; 4 4 -1; 4 4 -1; 4 4 4]
  
    edges = get_edges_ary(nodes_ary)
    node_bool_arys = get_node_bool_arys(nodes_ary)
    node_dims_arys = get_node_dims_arys(nodes_ary)  
    history = Int[]

    board =  cat(node_dims_arys, node_bool_arys, dims=3)
    
    return GameEnv(edges, node_dims_arys, node_bool_arys,  Int64[], trues(maximum(edges)), false, history)

end

#Don't know how to update action_mask, given a path

#function GI.set_path!(env::GameEnv, path)
#    env.history = nothing
#    env.path = path
#
#    update_actions_mask!(env)
#    any(env.amask) || (env.finished = true)
#  end
function GI.set_state!(env::GameEnv, state)
    #print("\n \n set state \n \n")
    env.edges = state.edges
    env.history = state.history
    env.node_dims_arys= state.node_dims_arys
    env.node_bool_arys = state.node_bool_arys
    env.finished = state.finished
    env.reward_list = state.reward_list
    env.amask = state.amask
    return
end

function GI.clone(env::GameEnv)
    #print("\n \n CLONED \n \n")
    history = isnothing(env.history) ? nothing : copy(env.history)
    return GameEnv(copy(env.edges), copy(env.node_dims_arys), copy(env.node_bool_arys), copy(env.reward_list), copy(env.amask), env.finished, history)
end

GI.two_players(::GameSpec) = false
 

GI.actions(::GameSpec) = collect(1:6)



history(env::GameEnv) = env.history

#####
##### Defining game rules
#####


function update_actions_mask!(env::GameEnv, action)
    env.amask[action] = false
end

GI.actions_mask(env::GameEnv) = env.amask

# Update the game status
function update_status!(env::GameEnv, action)
    
    update_actions_mask!(env, action)
    env.finished = !any(env.amask)

    true
end
  
function GI.play!(env::GameEnv, action)
    isnothing(env.history) || push!(env.history, action)

    update_status!(env, action)
  

    node_i0, node_i1 = first.(Tuple.(findall( x -> x == action, env.edges)))

    n0 = env.node_dims_arys[node_i0]
    n1 = env.node_dims_arys[node_i1]

    n0_B = env.node_bool_arys[node_i0]
    n1_B = env.node_bool_arys[node_i1]

    if_diff = !(n0_B[node_i1])
    if if_diff
        
        contract_dims = n0 .+ n1
        contract_dims[n0.*n1 .!= 0 ] .=n0[n0.*n1 .!= 0 ] .* n1[n0.*n1 .!= 0 ]
        
        contract_bool = n0_B .| n1_B

        flop = prod(filter(x -> x != 0, n0[.!contract_bool]))*prod(filter(x -> x != 0, n1[.!contract_bool]))*prod(filter(x -> x != 0, n0[contract_bool]))
        
        
        contract_dims[contract_bool] .= 0  
        
        env.node_dims_arys[contract_bool] .= [contract_dims]
        env.node_bool_arys[contract_bool] .= [contract_bool]

        push!(env.reward_list, -flop)
    end
   
end


  
    


  
GI.current_state(env::GameEnv) = (node_dims_arys = copy(env.node_dims_arys), node_bool_arys = copy(env.node_bool_arys), edges = copy(env.edges), history = copy(env.history), finished = copy(env.finished), amask = copy(env.amask), reward_list = copy(env.reward_list))
  
GI.white_playing(env::GameEnv) = true




#####
##### Reward shaping
#####



function GI.game_terminated(env::GameEnv)
    #print(all(all(x -> x == true, vec) for vec in env.node_bool_arys))
    return all(all(x -> x == true, vec) for vec in env.node_bool_arys)
end

  
  
function GI.white_reward(env::GameEnv)
    #print("white_reward")
  R = sum(env.reward_list)
  
    return R
end


function GI.vectorize_state(::GameSpec, state)
  return convert(Array{Float32}, cat(stack(state.node_dims_arys, dims=1), stack(state.node_bool_arys, dims=1), dims =3))
end  







GI.action_string(::GameSpec, a) = string(a)

function GI.parse_action(g::GameSpec, str)
    try
      p = parse(Int, str)
      1 <= p <= 6 ? p : nothing
    catch
      nothing
    end
  end

function GI.render(env::GameEnv, with_position_names=true, botmargin=true)
    
    print("\n BOOL ARRAY: \n")
    print(env.node_bool_arys)
    print("\n DIM ARRAY: \n")
    print(env.node_dims_arys)
    print("\n AMASK: \n")
    print(env.amask)
    print("\n REWARD LIST: \n")
    print(env.reward_list)

    botmargin && print("\n")
  end