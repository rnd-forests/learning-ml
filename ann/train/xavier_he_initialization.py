"""
Xavier and He Initialization
----------------------------
To prevent vanishing/exploding gradients problems, we need to control the signal flow properly in both direction:
forward (to make predictions), backward (when backpropagating gradients):
-> the variance of the outputs of each layer should be equal to the variance of its inputs
-> the gradients should have equal variance before and after flowing through a layer in the reverse direction


"""
