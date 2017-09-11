"""
Tweaking, Dropping, or Replacing the upper layers
-------------------------------------------------
The output layer and upper layers of the original model are not likely useful for the new model used
in our new task. Therefore, we should find the right number of hidden layers to reuse.

Freezing all copied layers first and observing model's performance. We can then try to unfreeze one or two
of the top hidden layers to let backpropagation tweak their weights, and make a comparision with the old
performance.

The more training instances we have, the more layers we can unfreeze.


Model Zoos
----------
Ref: https://github.com/tensorflow/models
Ref: https://github.com/tensorflow/models/tree/master/slim
Ref: https://github.com/BVLC/caffe/wiki/Model-Zoo
Ref: https://github.com/ethereon/caffe-tensorflow (Caffe to Tensorflow converter)


Unsupervised Pretraining
------------------------
General idea: using unsupervised algorithms to train each layer one by one. Each layer is trained using the
              output of the previous layer (except the frozen one). After every single layer is trained, we
              can fine-tune the network using supervised learning.

Example: If we don't have much labeled training data, we can firstly try to gather more labed data.
         If the process of collecting labeled data takes too long or too expensive, we can try to
         train each layer one by one, starting with the lowest layer, using  an unsupervised feature
         dectector algorithm such as RMBs (Restricted Boltzmann Machines). After each layer is trained
         using unsupervised learning, we can try to use supervised algorithm (i.e., backpropagation algorithm)
         to tweak the network.

Ref: https://en.wikipedia.org/wiki/Restricted_Boltzmann_machine


Pretrain on an Auxiliary Task
-----------------------------
General idea: training a first neural network on an auxiliary task for which we can easily obtain labeled data.
              After that, we can reuse the lower layers of that network for your actual task.

Keywords: Max Margin Learning
"""
