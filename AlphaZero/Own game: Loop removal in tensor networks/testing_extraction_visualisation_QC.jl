using AlphaZero                                                                 # Reinforcement Learning Package
using Tenet                                                                     # TensorNetwork Package
using Graphs                                                                    # Nice and efficienct way of representing the connectivity in the tensor_network
using GraphPlot                                                                 # Graph visualisation package
using FileIO
using JLD2

include("julia_functions.jl")                                                   # Paste a copy of julia_functions.jl inside of the directory
include("game_v10_10x10_QC.jl")
include("params1.jl")


# Then, pass it to the experiment initialization:
experiment = AlphaZero.Experiment("QC_2x6_d5", GameSpec(), params, Network, netparams, benchmark)
#AlphaZero.Scripts.test_game(experiment, n=1)
println("Test passed!!! Starting AlphaZero environment")
#AlphaZero.Scripts.explore(experiment)
#Scripts.dummy_run(experiment)
#AlphaZero.Scripts.train(experiment)

#  AlphaZero Player
Ses = Session(experiment)
p = AlphaZeroPlayer(Ses.env)
gspec = Ses.env.gspec

trace = play_game(gspec, p; flip_probability=0.)

print("\n\n\n_____   The agent has reached the final position  ___________\n")
taken_path = last(trace.states).history
taken_path = [action for action in taken_path]
println("Respective loop lengths = ", [length(taken[1]) for taken in taken_path])


rewards_list = last(trace.states).reward_list
reward =  sum(rewards_list)
println("The obtained rewards throughout the AlphaZero path = ", rewards_list, reward)


# RandomPlayer
p = AlphaZero.RandomPlayer()
gspec = Ses.env.gspec

trace = play_game(gspec, p; flip_probability=0.)

print("\n\n\n_____   The agent has reached the final position  ___________\n")
taken_path = last(trace.states).history
taken_path = [action for action in taken_path]
println("Respective loop lengths = ", [length(taken[1]) for taken in taken_path])


rewards_list = last(trace.states).reward_list
reward =  sum(rewards_list)
println("The obtained rewards throughout the random path = ", rewards_list, reward)

## Trying to extract_graph_representation of the initial state for visualisation purpose
adj_m = first(trace.states).current_adjacency
println("the length of the trace = ", length(trace.states))

# FileIO.save("transport variables/taken_path.jld2","taken_path", taken_path)

function visualise_gameplay(state_traces)

    
    #Function which makes a visualisation of the agent's taken path, this is done
    #based on the graph visualisation extracted from the traces and from the
    #history of the action list
    


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


function sized_adjacency_visualisation(state_traces)
    for (j, state) in enumerate(state_traces)
        println("State number = $j")
        display(state.weighted_edge_list)
    end
end




