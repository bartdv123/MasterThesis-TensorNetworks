using CSV, DataFrames

# Reading a CSV file
con_dat = CSV.read("test_connections.csv", DataFrame)
con = Matrix(con_dat)
nodes = zeros(Int, (size(con, 1), size(con, 2), 2))
nodes[:, :,1] = con

sizes_dat = CSV.read("test_sizes.csv", DataFrame)
sizes = Matrix(sizes_dat)
nodes[:, :,2] = sizes
