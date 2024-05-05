using AlphaZero                                                                 # Reinforcement Learning Package
using Tenet                                                                     # TensorNetwork Package
using Graphs                                                                    # Nice and efficienct way of representing the connectivity in the tensor_network
using GraphPlot                                                                 # Graph visualisation package


include("julia_functions.jl")                                                   # Paste a copy of julia_functions.jl inside of the directory
#include("game_v4_graph in env.jl")
#include("game_v4_graph not in env copy.jl")
include("game_v10_10x10_QC.jl")
include("params1.jl")


# Then, pass it to the experiment initialization:
experiment = AlphaZero.Experiment("QC", GameSpec(), params, Network, netparams, benchmark)
#AlphaZero.Scripts.test_game(experiment, n=1)
println("Test passed!!! Starting AlphaZero environment")
#AlphaZero.Scripts.explore(experiment)
#rScripts.dummy_run(experiment)
#AlphaZero.Scripts.train(experiment)
Ses = Session(experiment)
p = AlphaZeroPlayer(Ses.env)
gspec = Ses.env.gspec

trace = play_game(gspec, p; flip_probability=0.)

print("\n\n\n_____   The agent has reached the final position  ___________\n")
taken_path = last(trace.states).history
println("The path taken by the agent is = ", taken_path)
rewards_list = last(trace.states).reward_list
reward =  sum(rewards_list)
println("The obtained rewards throughout the path = ", rewards_list)

### Trying to extract_graph_representation of the initial state for visualisation purpose
adj_m = first(trace.states).current_adjacency
println("the length of the trace = ", length(trace.states))
# the graph representation can be extracted from the trace.states!!!


function visualise_gameplay(state_traces)

    """
    Function which makes a visualisation of the agent's taken path, this is done
    based on the graph visualisation extracted from the traces and from the
    history of the action list
    """


    old_graph = Graphs.SimpleGraphs.SimpleGraph(first(state_traces).current_adjacency)
    println(length(state_traces[2:end]))
    taken_path = last(state_traces).history



    for (j, state) in enumerate(state_traces[2:end])

        # step one: visualisation of the action on old_graph
        choosen_cycle = taken_path[j][1]
        choosen_edge = taken_path[j][2]
        global locs_x
        global locs_y
        if j == 1
            layout = spectral_layout(old_graph)
            locs_x = layout[1]
            locs_y = layout[2]
            display_selected_action_QC(old_graph, choosen_cycle, choosen_edge, locs_x, locs_y)
        end
        if j > 1 
            display_selected_action_QC(old_graph, choosen_cycle, choosen_edge, locs_x, locs_y)
        end
        # step two: visualisation of the obtained new_graph
        new_graph = Graphs.SimpleGraphs.SimpleGraph(state.current_adjacency)
        
        # the new_graph becomes the old_graph for the next action
        old_graph = new_graph
    end
        
end

visualise_gameplay(trace.states)

function sized_adjacency_visualisation(state_traces)
    for (j, state) in enumerate(state_traces)
        println("State number = $j")
        display(state.weighted_edge_list)
    end
end

#sized_adjacency_visualisation(trace.states)




