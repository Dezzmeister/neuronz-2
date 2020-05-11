# Neuronz-2

A deep learning library that lets you train and save neural networks with multiple hidden layers of various types.
Example code for both architectures can be found in Neuronz-2/src/examplecode.


## Old Architecture (only supports fully connected layers):

1. Start by creating or loading a `Network` (from a serialized `Network` file)
2. Write code to load your data as input `Vectors` and associated ideal output `Vectors`
3. Pack this data into associated `Vector[]s`
4. Write a function (`OutputVerificationScheme`) to evaluate the success of one run. This function is not used to train the network, only to calculate the success of an epoch.
5. Write a function (`LearningRateAdjuster`) to set the learning rate based on the current learning rate, the current epoch, and the previous success rate
6. Create a `NetworkRunner` to train the network


## New Architecture (supports layers of any type):

The new architecture uses the `Layer` interface, which has an input tensor type and an output tensor type. You can specify a Layer that accepts a Vector and returns a Vector, or one that accepts a Vector and returns a Tensor3, for instance. You can also connect several layers together (for instance, using a `LayerSequence`), and the connected sequence is indistinguishable from a single layer.

Several layers are already implemented (example: `DenseLayer`, `ConvolutionLayer2`, `PoolingLayer`). You can construct your own layers by implementing the `Layer` interface: you need to define what happens when data is fed forward through your layer, and you need to define what happens when the error gradient is propagated back through your layer. You will also need to specify how many learnable parameters your layer has, and how many layers your layer is composed of.

**IMPORTANT**: Each layer of a network MUST be a separate `Layer` object. For example, you cannot create one instance of a sigmoid layer and use it more than once in your network; each sigmoid layer must be a different instance. This is because the `Layer` interface is designed so that layers will most likely need to store some internal state related to their previous input or output. This is done so that gradients can be calculated correctly after a forward pass: the layer needs to remember some information about the previous forward pass.

### Activation Layers

There are two types of activation layers: `ElementActivationLayer` and `TensorActivationLayer`. `ElementActivationLayer` is used for activation functions that are applied element-wise to the input tensor. The sigmoid and ReLU activation functions are implemented in this way.

`TensorActivationLayer` is used for activation functions that accept a tensor input and return an output with the same shape. These functions are applied to the entire tensor at once. For example, the softmax activation function is implemented in this way, because it needs to sum the elements of the input Vector.


## Parallel Architecture (Temporary)
**NOTE: The parallel architecture is a temporary hack to improve training times until GPU optimizations are finished.**

The parallel architecture exploits multithreading and the parallel nature of mini-batch gradient descent to accelerate network training. A layer using the parallel architecture must implement `ParallelLayer`, which extends `Layer`. The `ParallelLayer` interface specifies three parallel versions of the methods specified in `Layer`. The difference is that the parallel methods must not modify the state of the layer, with the exception of `parallelUpdate()`. 

The `parallelForwardPass()` method returns an object containing inputs and outputs for each layer in the network. A layer implementing this method must create a `ParallelForwardPass` object (or add to a previous one, from a previous layer for instance). This object maps Layer objects to input and output tensors, so that a layer can use itself as a key to obtain its latest input and output.

The `parallelBackprop()` method returns an object containing gradients for each layer. The result of the previous forward pass is given to the method so that layers can access their most recent inputs and outputs. This method returns a `ParallelBackwardPass` object, which maps Layers to their gradients. Gradients are provided as a list of tensor types, and layers can calculate gradients and put them in the object using their own instances as a key (just like `ParallelForwardPass`).

The `parallelUpdate()` method accepts the result of a previous backward pass. Layers can update their state in this function, because this is intended to be called after calculating all gradients for one mini-batch. If doing mini-batch SGD, several backward passes may need to be combined into one. This can be done by summing all the gradients associated with a given layer, then dividing by the size of one mini-batch.


## GPU Optimizations (WIP)

GPU optimizations using OpenCL kernels and BLAS routines are a work in progress. This should significantly increase training speeds.
