"""
It's not a good idea to train large DNN from scratch. We should find an existing NN that does the similar task
to the one that we're trying to accomplish. Then, we reuse the lower layers of that network.

This technique is called TRANSFER LEARNING. It speeds up training process and requires much less training data.

Transfer learning will only work well if the inputs have similar low-level features.

Reusing a TensorFlow model
--------------------------

"""
