using AlphaZero                                                                 # Reinforcement Learning Package
using Tenet                                                                     # TensorNetwork Package
using Graphs                                                                    # Nice and efficienct way of representing the connectivity in the tensor_network
using GraphPlot                                                                 # Graph visualisation package


include("julia_functions.jl")                                                   # Paste a copy of julia_functions.jl inside of the directory
#include("game_v4_graph in env.jl")
#include("game_v4_graph not in env copy.jl")
include("game_v5_generate_locally.jl")
include("params.jl")


"""
Part to create a Tenet.TensorNetwork to try and play a game on
--> Initial testing using the Frucht graph structure with all bond dimensions 3
"""

# Then, pass it to the experiment initialization:
experiment = AlphaZero.Experiment("Loopremoval", GameSpec(), params, Network, netparams, benchmark)
#AlphaZero.Scripts.test_game(experiment, n=1)

#AlphaZero.Scripts.explore(experiment)
Scripts.dummy_run(experiment)
