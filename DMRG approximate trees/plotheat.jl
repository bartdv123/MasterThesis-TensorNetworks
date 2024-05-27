using Plots

# Let's assume these are your matrix and labels
matrix = rand(26, 26)
x_labels = 1:26;
y_labels = 1:26;

# Create a heatmap
p=heatmap(x_labels, y_labels, matrix, c=:coolwarm, xlabel="Maximal bond dimension", ylabel="Loop removal step")

# Show the plot
display(p)