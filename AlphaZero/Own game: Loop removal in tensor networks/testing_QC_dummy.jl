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
experiment = AlphaZero.Experiment("QC_8x7", GameSpec(), params, Network, netparams, benchmark)
#AlphaZero.Scripts.test_game(experiment, n=1)
println("Test passed!!! Starting AlphaZero environment")
AlphaZero.Scripts.explore(experiment)
#Scripts.dummy_run(experiment)
#AlphaZero.Scripts.train(experiment)
