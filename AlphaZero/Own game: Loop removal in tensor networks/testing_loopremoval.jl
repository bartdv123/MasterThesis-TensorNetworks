using AlphaZero                                                                 # Reinforcement Learning Package
using Tenet                                                                     # TensorNetwork Package
using Graphs                                                                    # Nice and efficienct way of representing the connectivity in the tensor_network
using GraphPlot                                                                 # Graph visualisation package



include("julia_functions.jl")                                                   # Paste a copy of julia_functions.jl inside of the directory
include("game.jl")
include("params.jl")



"""
Part to create a Tenet.TensorNetwork to try and play a game on
--> Initial testing using the Frucht graph structure with all bond dimensions 3
"""

#TODO: Fill function where bondsize is randomized and sampled from a distribution. 

dimension = 3                                                                   # Update this to allow sized adjacency extraction
G = Graphs.smallgraph(:frucht)
tn1 = fill_with_random(G, dimension, true)


# Then, pass it to the experiment initialization:
experiment = AlphaZero.Experiment("Loopremoval", GameSpec(), params, Network, netparams, benchmark, init_args = (tn1,))

Scripts.train(experiment)
