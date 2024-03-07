using AlphaZero                                                                 # Reinforcement learning package
import AlphaZero.GI                                                             # Game interface
using Graphs                                                                    # Nice and efficienct way of representing the connectivity in the tensor_network
using GraphPlot                                                                 # Graph visualisation package
using Tenet                                                                     # TensorNetwork Package

include("julia_functions.jl")                                                   # Paste a copy of julia_functions.jl inside of the directory

"""
Updated version 05 with locally generated graphs 
used for the logic of edge selection and cost function calculations
"""

"""
Setting up a game environment and game structure -> try to do initial play and
testing on the Frucht Graph
"""


struct GameSpec <: GI.AbstractGameSpec end                                      # Create a AbstractGameSpec structure


mutable struct GameEnv <: GI.AbstractGameEnv                                    # Create a mutable strucutre -> this is updated during gameplay

    #TODO: Simplify this GameEnv based on removing double stored things:

    # Remove list_of_edges and work with the sized_edges instead.
    # Amask masks out the sized edges.

    sized_adjacency::Matrix{Int64}                                              # Sizes of the edges inside an adjecancy matrix
    current_adjacency::Matrix{Int64}                                            # Current adjacency representation of the underlying graph. => update when action is taken
    boolean_action::Matrix{Int64}                                               # Current available_actions, parametrized as a matrix of (cycle ₓ edge in cycle)
    list_of_edges::Vector{Any}                                                  # List of edges within the graph
    cycle_basis::Vector{Any}                                                    # List of cycles current within the graph
    weighted_edge_list::Vector{Any}                                             # Static List of edges and dimensionality [[source, drain, size], ...]
    reward_list::Array{Int64}                                                   # The rewards the agent got for the choices it made during the gameplay
    amask::BitVector                                                            # Boolean_edge_availability, which edges within list_of_edges are still available
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

    dimension = 3                                                               # Update this to allow sized adjacency extraction
    G = Graphs.smallgraph(:frucht)
    TN = fill_with_random(G, dimension, false, true)

    graph, tv_map, ie_map, weighted_edge_list, ei_map = extract_graph_representation(TN, false) # Extract the graphs.jl structure from the Tenet.TensorNetwork
    sized_adjacency = sized_adj_from_weightededges(weighted_edge_list, graph)
    initial_adjacency = adjacency_matrix(graph)
    cycle_basis = minimum_cycle_basis(graph)
    list_of_edges = cycle_basis_to_edges(cycle_basis)
    boolean_action = create_actionmatrix(graph, list_of_edges)
    history = []
    amask = update_edge_availability(graph, weighted_edge_list, trues(length(weighted_edge_list)))

    return GameEnv(
        sized_adjacency,
        initial_adjacency,
        boolean_action,
        list_of_edges,
        cycle_basis,
        weighted_edge_list,
        Int64[],
        amask,
        false,
        history    
        )

end


function GI.set_state!(env::GameEnv, state)

    """
    In place modification of a state
    => No need to copy
    """

    env.sized_adjacency = state.sized_adjacency
    env.current_adjacency = state.current_adjacency
    env.boolean_action = state.boolean_action
    env.list_of_edges = state.list_of_edges
    env.cycle_basis = state.cycle_basis
    env.weighted_edge_list = state.weighted_edge_list
    env.reward_list = state.reward_list
    env.amask = state.amask
    env.finished = state.finished
    env.history = state.history

end


"""
The state returned by this function may be stored (e.g. in the MCTS tree) 
and must therefore either be fresh or persistent. If in doubt, you should make a copy.
--> Function below adresses this problems through deepcopying the environment variables 
"""


function GI.clone(env::GameEnv)

    """
    Return an independent copy of the given environment.
    => should use deepcopy just to be structure
    => add new pointers in computer memory
    """

    history = isnothing(env.history) ? nothing : deepcopy(env.history)

    return GameEnv(
    deepcopy(env.sized_adjacency),
    deepcopy(env.current_adjacency),
    deepcopy(env.boolean_action),
    deepcopy(env.list_of_edges),
    deepcopy(env.cycle_basis),
    deepcopy(env.weighted_edge_list),
    deepcopy(env.reward_list),
    deepcopy(env.amask),
    deepcopy(env.finished),
    deepcopy(env.history)
    )

end


GI.two_players(::GameSpec) = false


GI.actions(::GameSpec) = collect(1:18)                                          # 18 edges in a frucht graph
     

function GI.available_actions(env::GameEnv)

    # Returns a list of tuples where the boolean action mask == 1
    ones_indices = findall(x -> x == 1, env.boolean_action)
  
    #  Convert the indices to tuples of (row, column) format
    indices =  [(i[1], i[2]) for i in ones_indices]
    return indices 
end


history(env::GameEnv) = (env.history)                                           


"""
Defining basic game rules
What does it mean to play a game???
"""



GI.actions_mask(env::GameEnv) = (env.amask)


# Update the game status when performing an action              
function update_status!(env::GameEnv, action)

    """
    Updates the game status INPLACE after an action is performed!
    Based on the new current_graph_representation which is 
    extracted from current_adjacency
    """

    update_action_mask!(env, action)


end


function update_action_mask!(env::GameEnv, action)                              # Mask for action which are not possible anymore after performing an action

    """
    An inplace modification function for the action mask.
    An action is uniquely indentified and parametrized by the index inside of the boolean_edge_availability
    array. This way, performing this action corresponds to masking this specific
    action by a false inside of the amask.
    """

    # Generate the current graph structure based on the adjacency matrix
    current_graph_representation = Graphs.SimpleGraphs.SimpleGraph(env.current_adjacency)
    env.cycle_basis = minimum_cycle_basis(current_graph_representation)
    env.boolean_action = create_actionmatrix(current_graph_representation, env.list_of_edges)
    




    show = true
    

    # Updating the edge mask based on the currently present loops inside of the graph
    env.amask = update_edge_availability(current_graph_representation, env.weighted_edge_list, env.amask)
    # Checking if finishing condition is reached, edge availability is empty if the grapph is a tree
    if env.amask == zeros(length(env.amask))
        env.finished = true
        if show == true
            #display the final tree that the network found
            nodes = [node for node in vertices(current_graph_representation)]
            locs_x =     [4, 4, -5, -2, 0, 0, 2, 0, -3, -1, -6, -4]
            locs_y =  -1*[-2, 1, -2, -1, 0, -2, 0, 3, 3, 1, 1, 0]
            display(gplot(current_graph_representation, locs_x, locs_y, nodelabel=nodes, nodefillc=colorant"springgreen3")) 
        end
    end
end



function GI.play!(env::GameEnv, action)

    """
    What should happen in the game state when the agent takes an action
    -> Should happen inplace?
    """
    #visualisation of what is being fed into the play! function
    # TODO: Implement the possibility of updating the state spaces when
    # performing an action.
    choosen_cycle = env.cycle_basis[action[1]]
    choosen_edge = env.list_of_edges[action[2]]


    isnothing(env.history) || push!(env.history, (choosen_cycle, choosen_edge))                        
    
    # Generate the current graph structure: before edge removal through DMRG
    old_graph = Graphs.SimpleGraphs.SimpleGraph(env.current_adjacency)
    
    # Remove the edge from the graph structure: update current_adjacency

    env.current_adjacency[choosen_edge[1], choosen_edge[2]] = 0
    env.current_adjacency[choosen_edge[2], choosen_edge[1]] = 0
    
    """
    Rewards while_playing
    """

    #TODO: IMPLEMENTATION OF DIFFERENT REWARD FUNCTIONS?
    # Reward function right now is a chi_max^5 for all chi inside of the MPS
    # structure which is uncovered after cutting an edge.
    # display_selected_action(old_graph, choosen_cycle, choosen_edge)
    # minimize the loss -> -1*
    push!(env.reward_list, -1*calculate_DMRG_cost(old_graph, env.weighted_edge_list, choosen_cycle, choosen_edge))

    env.weighted_edge_list = edge_weights_update_DMRG_exact(old_graph, choosen_cycle, choosen_edge, env.weighted_edge_list)
    env.sized_adjacency = sized_adj_from_weightededges(fully_weighted_edge_list, old_graph)
    # Update the other game variables such as edge availability based on cycle finding
    update_status!(env, action)                                                 # Updates the status of the amask, and game game_terminated status
    # --> Generates new possible cycle_basis -> cutting an edge can create new 
    # possible faces. 

   
end

# Some more neccesary implementations

GI.current_state(env::GameEnv) = 

(
sized_adjacency = deepcopy(env.sized_adjacency),
current_adjacency = deepcopy(env.current_adjacency),
boolean_action = deepcopy(env.boolean_action),
list_of_edges = deepcopy(env.list_of_edges),
cycle_basis = deepcopy(env.cycle_basis),
weighted_edge_list = (env.weighted_edge_list), 
reward_list = deepcopy(env.reward_list),
amask = deepcopy(env.amask),
finished = deepcopy(env.finished),
history = deepcopy(env.history)
)


GI.white_playing(env::GameEnv) = true


GI.action_string(::GameSpec, action) = string(action)


GI.heuristic_value(env::GameEnv) = Float64(sum(env.reward_list))


function GI.game_terminated(env::GameEnv)
    return env.finished
end


function GI.vectorize_state(::GameSpec, state)
    return convert(Array{Float32}, cat(stack(state.sized_adjacency, dims=1), stack(state.current_adjacency, dims=1), dims=3))
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

