# neuronz-2

A rudimentary library (?) that lets you train and save neural networks with multiple hidden layers (deep neural networks).
Uses a vectorized version of backpropagation to train networks. 

I wrote this to get a better understanding of the math.


## How to use this:

1. Start by creating or loading a `Network` (from a serialized `Network` file)
2. Write code to load your data as input `Vectors` and associated ideal output `Vectors`
3. Pack this data into associated `Vector[]s`
4. Write a function (`OutputVerificationScheme`) to evaluate the success of one run. This function is not used to train the network, only to calculate the success of an epoch.
5. Write a function (`LearningRateAdjuster`) to set the learning rate based on the current learning rate, the current epoch, and the previous success rate
6. Create a `NetworkRunner` to train the network
