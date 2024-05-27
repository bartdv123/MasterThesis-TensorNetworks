using AlphaZero                                                                 # Reinforcement Learning Package
using Tenet                                                                     # TensorNetwork Package
using Graphs                                                                    # Nice and efficienct way of representing the connectivity in the tensor_network
using GraphPlot                                                                 # Graph visualisation package
using Colors


include("julia_functions.jl")                                                   # Paste a copy of julia_functions.jl inside of the directory
#include("game_v4_graph in env.jl")
#include("game_v4_graph not in env copy.jl")
include("Final_discussed_toy_model.jl")
include("params.jl")


# Then, pass it to the experiment initialization:
experiment = AlphaZero.Experiment("Toymodel", GameSpec(), params, network, netparams, benchmark)
#Scripts.dummy_run(experiment)
#Dummy run worked! Lets train
AlphaZero.Scripts.train(experiment)
