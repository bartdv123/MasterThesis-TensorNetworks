using AlphaZero                                                                 # Reinforcement Learning Package
using Tenet                                                                     # TensorNetwork Package
using Graphs                                                                    # Nice and efficienct way of representing the connectivity in the tensor_network
using GraphPlot                                                                 # Graph visualisation package
using Colors

function visualise_gameplay(state_traces)

    """
    Function which makes a visualisation of the agent's taken path, this is done
    based on the graph visualisation extracted from the traces and from the
    history of the action list
    """


    old_graph = Graphs.SimpleGraphs.SimpleGraph(first(state_traces).current_adjacency)
    taken_path = last(state_traces).history



    for (j, state) in enumerate(state_traces[2:end])

        # step one: visualisation of the action on old_graph
        choosen_cycle = taken_path[j][1]
        choosen_edge = taken_path[j][2]
        display_selected_action_toyfinal(old_graph, choosen_cycle, choosen_edge)

        # step two: visualisation of the obtained new_graph
        new_graph = Graphs.SimpleGraphs.SimpleGraph(state.current_adjacency)
        nodes = [node for node in vertices(new_graph)]
        colors_for_edges = [colorant"grey" for edge in edges(new_graph)]

        #Planar representation
        locs_x = [0.8235890524005085, 1.0, 0.07580751318974355, -0.017032037929692323, -0.2944683465563266, 0.19505501947920223, -0.34666814155572356, -0.30642007335610566, 0.4977523253682399, -0.7358328318661933, -0.4535100245888959, -0.6351469771061515, -1.0, -0.8649522531330216, 0.7274677388545647, 0.8296686177085972, 0.4719036541802202, 0.698920176497434, 0.10082819193961834, 0.49215115665455755]
        locs_y = [-0.10897958507558192, 0.46721962406333617, 0.6797683283018006, -0.3196512600723005, -1.0, 0.13569820340952843, 0.38946479275835033, -0.2152231239505723, -0.5186006246689268, 0.20782072859317902, -0.55361343309079, -0.8258232353023699, 0.08322844321138567, -0.30754574543717594, 0.29675288939110867, 0.9916952245291406, 1.0, 0.6732513753667592, -0.8050366612343887, -0.13189261036750358]
        display(gplot(new_graph, locs_x, locs_y, edgestrokec = colors_for_edges, nodelabel=nodes, nodefillc=colorant"orange"))

        # the new_graph becomes the old_graph for the next action
        old_graph = new_graph
    end
        
end

include("julia_functions.jl")                                                   # Paste a copy of julia_functions.jl inside of the directory
#include("game_v4_graph in env.jl")
#include("game_v4_graph not in env copy.jl")
include("Final_discussed_toy_model.jl")
include("params1.jl")


# Then, pass it to the experiment initialization:
experiment = AlphaZero.Experiment("Toymodel", GameSpec(), params, Network, netparams, benchmark)
#AlphaZero.Scripts.test_game(experiment, n=1)
println("Tests passed. Starting AlphaZero environment")
#AlphaZero.Scripts.explore(experiment)
#rScripts.dummy_run(experiment)
#AlphaZero.Scripts.train(experiment)
Ses = Session(experiment)
p = AlphaZeroPlayer(Ses.env)
gspec = Ses.env.gspec

println("-------------------   ALPHAZERO TRAINED PLAYER   --------------------")
trace = play_game(gspec, p; flip_probability=0.)

taken_path = last(trace.states).history
println("The path taken by the agent is = ", taken_path)
println("Respective loop lengths = ", [length(taken[1]) for taken in taken_path])

rewards_list1 = last(trace.states).reward_list
reward1 =  sum(rewards_list1)
println("The obtained rewards throughout the path = ", rewards_list1)
println("The total cost throughout the path = ", reward1)


### Trying to extract_graph_representation of the initial state for visualisation purpose
adj_m = first(trace.states).current_adjacency
println("the length of the trace = ", length(trace.states))
# the graph representation can be extracted from the trace.states!!!
#visualise_gameplay(trace.states)



# RandomPlayer
println("-----------------------------   RANDOM PLAYER  -----------------------")
p = AlphaZero.RandomPlayer()
gspec = Ses.env.gspec

trace = play_game(gspec, p; flip_probability=0.)

taken_path = last(trace.states).history
taken_path = [action for action in taken_path]
println("Respective loop lengths = ", [length(taken[1]) for taken in taken_path])


rewards_list2 = last(trace.states).reward_list
reward2 =  sum(rewards_list2)
println("The obtained rewards throughout the random path = ", rewards_list2)
println("The total cost throughout the path = ", reward2)

## Trying to extract_graph_representation of the initial state for visualisation purpose
adj_m = first(trace.states).current_adjacency
println("the length of the trace = ", length(trace.states))

#visualise_gameplay(trace.states)

println("Relative cost Trained vs RandomPlayer = ", reward1/reward2)


