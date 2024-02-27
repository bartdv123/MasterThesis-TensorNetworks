using AlphaZero                                                                 # Reinforcement learning package
import AlphaZero.GI                                                             # Game interface
using Graphs                                                                    # Nice and efficienct way of representing the connectivity in the tensor_network
using GraphPlot                                                                 # Graph visualisation package
using Tenet                                                                     # TensorNetwork Package

include("julia_functions.jl")                                                   # Paste a copy of julia_functions.jl inside of the directory


"""
Updated version 04 with weighted edges, sized adj_matrix and boolean actions
for edge selection
"""

"""
Part to create a Tenet.TensorNetwork to try and play a game on
--> Initial testing using the Frucht graph structure with all bond dimensions 3
"""

#TODO: Fill function where bondsize is randomized and sampled from a distribution. 


"""
Setting up a game environment and game structure -> try to do initial play and
testing on the Vrucht Graph
"""
struct GameSpec <: GI.AbstractGameSpec end                                      # Create a AbstractGameSpec structure

"""
A game board will be related to the graph properties of
the underlying TensorNetwork. Extra properties such as edge dimensions are stored in an additional 
adjacency matrix like structure.
"""

mutable struct GameEnv <: GI.AbstractGameEnv                                    # Create a mutable strucutre -> this is updated during gameplay
    weighted_edge_list::Vector{Any}                                             # List of edges and dimensionality [[source, drain, size], ...]
    sized_adjacency::Matrix{Int64}                                              # Sizes of the edges inside an adjecancy matrix
    reward_list::Array{Int64}                                                   # The rewards the agent got for the choices it made during the gameplay
    amask::BitVector                                                            # Used by external solvers to interpret game position -> same as boolean_edge_availability
    finished::Bool                                                              # Boolean to represent if the game is finished
    history:: Union{Nothing, Vector}                                            # History of actions

end

GI.spec(::GameEnv) = GameSpec()

function GI.init(::GameSpec)

    """
    Initialisation of a game environment is done by extracting the relevant data
    from the graph representation of a Tenet.TensorNetwork
    Return an initialized GameEnv
    """

    dimension = 3                                                             # Update this to allow sized adjacency extraction
    G = Graphs.smallgraph(:frucht)
    TN = fill_with_random(G, dimension, false, true)

    graph, tv_map, ie_map, fully_weighted_edge_list, ei_map = extract_graph_representation(TN, false) # Extract the graphs.jl structure from the Tenet.TensorNetwork
    sized_connections = sized_adj_from_weightededges(fully_weighted_edge_list, graph)
    history = Int[]
    return GameEnv(fully_weighted_edge_list, sized_connections, Int64[], trues(length(fully_weighted_edge_list[1:10])), false, history)
end

function GI.set_state!(env::GameEnv, state)

    """
    In place modification of a state
    => No need to copy
    """

    env.weighted_edge_list = state.weighted_edge_list
    env.sized_adjacency = state.sized_adjacency
    env.reward_list = state.reward_list
    env.amask = state.amask
    env.finished = state.finished
    env.history = state.history

end

"""
The state returned by this function may be stored (e.g. in the MCTS tree) 
and must therefore either be fresh or persistent. If in doubt, you should make a copy.
--> Function below adresses this problems 
"""

function GI.clone(env::GameEnv)

    """
    Return an independent copy of the given environment.
    => should use deepcopy just to be structure
    => add new pointers in computer memory
    """

    history = isnothing(env.history) ? nothing : deepcopy(env.history)
    return GameEnv(deepcopy(env.weighted_edge_list), 
    deepcopy(env.sized_adjacency), 
    deepcopy(env.reward_list),
    deepcopy(env.amask),
    deepcopy(env.finished),
    deepcopy(history))

end


GI.two_players(::GameSpec) = false

GI.actions(::GameSpec) = collect(1:10)                                          # 18 edges in a frucht graph
                                         
function GI.available_actions(env::GameEnv)
    indices = Int[]
    #env.boolean_edge_availability = update_edge_availability(env.graph, env.weighted_edge_list, env.boolean_edge_availability)

    for i in eachindex(env.amask)
        if env.amask[i] == 1
            push!(indices, i)
        end
    end
    return indices 
end

history(env::GameEnv) = (env.history)                                           


"""
Defining basic game rules
What does it mean to play a game???
"""

function update_action_mask!(env::GameEnv, action)                              # Mask for action which are not possible anymore after performing an action

    """
    An inplace modification function for the action mask
    An action is uniquely indentified and parametrized by the index inside of the boolean_edge_availability
    array. This way, performing this action corresponds to masking this specific
    action by a false inside of the amask.
    """

    env.amask[action] = false
    if all(env.mask == true)
        env.finished = true
    end

end

GI.actions_mask(env::GameEnv) = (env.amask)


# Update the game status when performing an action              
function update_status!(env::GameEnv, action)

    """
    Updates the game status INPLACE after an action is performed!
    """
    update_action_mask!(env, action)

end


function GI.play!(env::GameEnv, action)

    """
    What should happen in the game state when the agent takes an action
    -> Should happen inplace?
    """
    
    #TODO: Implement the possibility of updating the state spaces when
    # performing an action.

    isnothing(env.history) || push!(env.history, action)                        
    
    # Based on the choosen action: integer between 1:length(edges)
    # --> Cut this specific edge
    selected_edge = env.weighted_edge_list[action]
    
    # Remove the edge from the graph structure
    # update the sized_adjacency matrix -> remove the edge
    env.sized_adjacency[selected_edge[1], selected_edge[2]] = 0
    env.sized_adjacency[selected_edge[2], selected_edge[1]] = 0

    update_status!(env, action)                                                 # updates the status of the amask, sized_adjacency, boolean_edge_availability, and game game_terminated

    """
    Rewards while_playing
    """
    println(length(env.reward_list))
    #TODO:

    # add the dimensionality of the severed index

    push!(env.reward_list, selected_edge[3])


end

# Some more neccesary implementations

GI.current_state(env::GameEnv) = (weighted_edge_list = (env.weighted_edge_list), 
sized_adjacency = (env.sized_adjacency),
reward_list = (env.reward_list),
amask = (env.amask),
finished = (env.finished),
history = (env.history))



GI.white_playing(env::GameEnv) = true

GI.action_string(::GameSpec, action) = string(action)
GI.heuristic_value(env::GameEnv) = Float64(sum(env.reward_list))

function GI.game_terminated(env::GameEnv)
    return env.finished
end

function GI.vectorize_state(::GameSpec, state)
    return convert(Array{Float32}, cat((state.sized_adjacency), dims =3))
end 

function GI.white_reward(env::GameEnv)
    obtained_rewards = sum(env.reward_list)
    return obtained_rewards
end





function GI.render(env::GameEnv, visualisation = false)

`   """
    What should happen when rendering a game environment
    """`

    print("\n SIZED ADJACENCY REPRESENTATION: \n")
    display(env.sized_adjacency)

    if visualisation == true
        current_graph_representation = env.graph
        nodes = [node for node in vertices(current_graph_representation)]
        display(gplot(current_graph_representation, nodelabel=nodes, nodefillc=colorant"springgreen3", layout=spring_layout))
    end

    print("\n ACTION MASK \n")
    println(env.amask)
    print("\n REWARD LIST: \n")
    println(env.reward_list)

end

    

### TESTING THE GAME INTERFACE IMPLEMENTATION

