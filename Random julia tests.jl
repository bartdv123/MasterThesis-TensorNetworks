using Tenet

# Create a vector
my_vector = [1.0, 2.0, 3.0]

# Convert the vector to a tensor with a named index
my_tensor = Tensor(my_vector, (:i,))

# Display the tensor
@show my_tensor