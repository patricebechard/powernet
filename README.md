# PowerNet

Implementation of the PowerNet

The model is pretty simple. The input is linked to a reservoir of fixed size using a fully connected layer. The Reservoir is a network where each if its nodes are connected to each other with various degrees of strength. The signal is propagated through the network for *n_steps* steps, which can only be represented as *n_steps* passes through the same linear layer. The signal is then outputted by the reservoir to the output layer using a fully connected layer.

We can see the model as recurrent neural network, where every node of the reservoir is linked to every other one.