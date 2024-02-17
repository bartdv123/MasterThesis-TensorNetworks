using Graphs
using GraphPlot
using Colors

g = SimpleGraph(5)
add_edge!(g, (1,2))
add_edge!(g, (2,3))
add_edge!(g, (3,4))
add_edge!(g, (4,1))
add_edge!(g, (1,5))
add_edge!(g, (1,3))
nodelabel = ["A", "B", "C", "D", "E"]
edgelabel = ["i", "j", "k", "l", "m", "n"]

# Set the size of the output image
size = (1600, 1200)

display(gplot(g, nodelabelsize=35, nodelabel=nodelabel, nodefillc= colorant"orange", edgelabel=edgelabel, edgelabeldistx=0.75, edgelabeldisty=1, layout=spring_layout))