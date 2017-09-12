"""
Tweaking neural network hyperparameters is a pretty tedious process because there're many of them to experiment with.
You can use any type of network topology (the way neurons are connected to other ones).
For simple MLP, we can tweak the number of layers, weight initialization logic, the number of neurons per layer, type
of activation function, etc.

Grid Search vs. Cross-validation can be used to find the best combination of hyperparameters; however, for the large
datasets and huge number of hyperparameters, this approach is too slow. Only a small part of the hyperparameter space
is explored in a reasonable amount of time.

We can use Randomized Search or tool like http://oscar.calldesk.ai instead.

For neural network, we can focus on a set of hyperparameters which are important to the performance of the network.

Number of Hidden Layers
-----------------------
In reality, MLP with one hidden layer can be used to model any complex functions !? if it has enough number of neurons.
We should experiment with neural network with 1 hidden layer first before adding more hidden layers
However, DNN can achieve the same results as shallow net (1 hidden layer) with 'exponentially' fewer neurons which, in
turn, makes the training process much faster.

Normally, realword data is structured in hierarchical way, so...
    - Lower hidden layers model low-level structures
    - Intermediate hidden layers combine low-level structures to model intermediate-level structures
    - And the same for highest hidden layers and output layer

-> We should start solving the problem with one or two hidden layers and observe the results.
-> Increase the number of hidden layers as the complexity of the problem increases until we start overfitting
   the training set.


Number of Neurons per Hidden Layer
----------------------------------
The number of neurons of the input and output layers is determined by the type of input and output your task requires.
As a result, we only need to adjust the number of neurons for hidden layers.
A common practice is to reduce the number neurons through each layer in the network because many low-level features
can coalesce (combine) into far fewer high-level features.

-> Increase the number of neurons for hidden layers until we start overfitting the training set.
-> We can start with a model with more layers and neurons more than necessary, and use some techniques like
   'early stopping', 'dropout' to prevent model from overfitting the training set. This is just a reverse
   of the above approach.


Activation Functions
--------------------
Most of the time, we use ReLU (or its variants) activation function for hidden layers because it's fast to compute.

For output layer, softmax activation function is generally used for classification tasks (if classes are mutually
exclusive). For regression task, no activation function is used in the output layer.
"""
