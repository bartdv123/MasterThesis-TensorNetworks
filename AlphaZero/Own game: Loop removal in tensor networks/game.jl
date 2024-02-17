using AlphaZero                                                                 # Reinforcement learning package
import AlphaZero.GI                                                             # Game interface
using Graphs                                                                    # Nice and efficienct way of representing the connectivity in the tensor_network
using GraphPlot                                                                 # Graph visualisation package
using Tenet                                                                     # TensorNetwork Package

include("julia_functions.jl")                                                   # Paste a copy of julia_functions.jl inside of the directory


"""
Part to create a Tenet.TensorNetwork to try and play a game on
--> Initial testing using the Frucht graph structure with all bond dimensions 3
"""

#TODO: Fill function where bondsize is randomized and sampled from a distribution. 

dimension = 3                                                                   # Update this to allow sized adjacency extraction
G = Graphs.smallgraph(:frucht)
tn1 = fill_with_random(G, dimension, true)


"""
Setting up a game environment and game structure -> try to do initial play and
testing on the Vrucht Graph
"""
struct GameSpec <: GI.AbstractGameSpec end                                      # Create a AbstractGameSpec structure

"""
A game board will be related to the graph properties of
the underlying TensorNetwork. The laplacian_matrix is used to visualise
connectivity and compute tree-ness.
Extra properties such as edge dimensions are stored in an additional 
adjacency matrix like structure
"""

mutable struct GameEnv <: GI.AbstractGameEnv                                    # Create a mutable strucutre -> this is updated during gameplay
    game_board_laplacian::Matrix{Int64}                                         # Defining the types to have fast and good code
    sized_adjacency::Matrix{Int64}                                              # Sizes of the edges inside an adjecancy matrix
    adjacency_matrix::Matrix{Int64}                                                    # Way for quick graph visualisation during gameplay
    num_loops::Int64                                                            # Amount of loops in the network
    cycles_list::Vector{Vector{Int64}}                                          # The specific list of cycles which are present in the network                                                               # The definitions of the simple cycles
    reward_list::Array{Int64}                                                   # The rewards the agent got for the choices it made during the gameplay
    action_mask::BitVector # BitVector is specialized type to use Booleans      # After cutting a loop inside of the network, the actions of cutting this loop in a different way should be masked out                             
    finished::Bool                                                              # Boolean to represent if the game is finished
    history:: Union{Nothing, Vector}                                            # History of actions

end

GI.spec(::GameEnv) = GameSpec()

function GI.init(::GameSpec, TN::Tenet.TensorNetwork)                           
    """
    Initialisation of a game environment is done by extracting the relevant data
    from the graph representation of a Tenet.TensorNetwork
    """

    graph, tensor_vertex_map, index_edge_map = extract_graph_representation(TN) # Extract the graphs.jl structure from the Tenet.TensorNetwork
    laplacian = laplacian_matrix(graph)
    sized_connections = dimension*adjacency_matrix(graph)
    adjacency_matrix = adjacency_matrix(graph)
    num_loops = length(cycle_basis(graph))
    cycles_list = cycle_basis(graph)
    history = Int[]

    return GameEnv(laplacian, sized_connections, adjacency_matrix, num_loops, cycles_list, Int64[], trues(num_loops), false, history)
end

function GI.set_state!(env::GameEnv, state)
    #print("\n \n set state \n \n")
    env.game_board_laplacian = state.game_board_laplacian
    env.sized_adjacency = state.sized_adjacency
    env.adjacency_matrix = state.adjacency_matrix
    env.num_loops = state.num_loops
    env.cycles_list = state.cycles_list
    env.reward_list = state.reward_list
    env.action_mask = state.action_mask
    env.finished = state.finished
    env.history = state.history
    return
end

"""
The state returned by this function may be stored (e.g. in the MCTS tree) 
and must therefore either be fresh or persistent. If in doubt, you should make a copy.
--> Function below adresses this problems 
"""

function GI.clone(env::GameEnv)
    #print("\n \n CLONED \n \n")
    history = isnothing(env.history) ? nothing : copy(env.history)
    return GameEnv(copy(env.game_board_laplacian), copy(env.sized_connections), 
    copy(env.adjacency_matrix), copy(env.num_loops), copy(env.cycles_list), 
    copy(env.reward_list), copy(env.action_mask), 
    copy(env.finished), copy(env.history))
end


GI.two_players(::GameSpec) = false                                              # It's a single player game!
 

GI.actions(::GameSpec) = collect(1:num_loops)                                   # Actions possible to the agent, ie. select an edge for cutting


history(env::GameEnv) = env.history                                             


"""
Defining basic game rules
What does it mean to play a game???
"""

function update_action_mask!(env::GameEnv, action)                              # Mask for action which are not possible anymore after performing an action
    env.action_mask[action] = false
end

GI.actions_mask(env::GameEnv) = env.action_mask


# Update the game status                                                        # Perform an action              
function update_status!(env::GameEnv, action)
    
    update_actions_mask!(env, action)
    env.finished = !any(env.amask)

    true
end


function GI.play!(env::GameEnv, action)
    """
    What should happen in the game state when the agent takes an action
    """

    #TODO: Implement the possibility of updating the state spaces when
    # performing an action. 




    """
    Rewards while_playing
    """

    #TODO: Implement a way of scoring the cost/reward of an action
    # taken by the agent



end

# Some more neccesary implementations

GI.current_state(env::GameEnv) = (game_board_laplacian = copy(env.game_board_laplacian), 
    sized_adjacency = copy(env.sized_adjacency), 
    adjacency_matrix= copy(env.adjacency_matrix), 
    num_loops = copy(env.num_loops), 
    cycles_list = copy(env.cycles_list), 
    reward_list = copy(env.reward_list), 
    action_mask = copy(env.action_mask), 
    finished = copy(env.finished), 
    history = copy(env.history))

GI.white_playing(env::GameEnv) = true

function GI.game_terminated(env::GameEnv)
    return env.finished
end

function GI.vectorize_state(::GameSpec, state)
    return convert(Array{Float32}, cat(state.game_board_laplacian, state.sized_adjacency, dims =3))
end 

function GI.white_reward(env::GameEnv)
    obtained_rewards = sum(env.reward_list)
    return obtained_rewards
end





function GI.render(env::GameEnv, visualisation = true)

`   """
    What should happen when rendering a game environment
    """`

    print("\n LAPLACIAN REPRESENTATION: \n")
    display(env.game_board_laplacian)

    if visualisation == true
        current_graph_representation = SimpleGraph(env.adjacency_matrix)
        nodes = [node for node in vertices(current_graph_representation)]
        display(gplot(current_graph_representation, nodelabel=nodes, nodefillc=colorant"springgreen3", layout=spring_layout))
    end

    print("\n SIZED ADJACENCY REPRESENTATION: \n")
    display(env.sized_adjacency)
    print("\n Action mask \n")
    display(env.action_mask)
    print("\n REWARD LIST: \n")
    print(env.reward_list)

end

    






